#!/usr/bin/env python3
"""One fixed single-pass Heston v0 transport circuit: range, fidelity, gate cost.

Run (from the SuperEncoder repository root):
  python heston_coninuity/single_pass_transport_circuit.py

FROZEN ARCHITECTURE: the 16 ordered, odd-Y Pauli strings with maximum
contiguous span 3 selected by the preceding finite trajectory experiment
(support_3_budget_16_trajectory). Selection was trained on v0 in [0.01, 0.20];
anything beyond this interval is an architectural extrapolation test.

Two angle prescriptions are evaluated for the SAME single-pass circuit:
  (1) FOLDED: integrate the independently fitted tangent coefficient c_j(v)
      from the reference to the endpoint and apply each exp(-i theta_j P_j) once.
      This approximates, but does NOT equal, the time-ordered evolution.
  (2) ENDPOINT FIT: locally optimise those 16 angles against target fidelity,
      warm-started from folded angles (and a nearby previous solution).
      Its extra classical optimisation cost is NOT included in gate metrics.
      It is a local numerical result, not a globally optimality certificate.

The circuit is compiled to H/S/Sdg/RZ/CX with nearest-neighbour routing and
also all-to-all connectivity. The gate list is verified by direct statevector
simulation at exported checkpoints. Neither circuit includes preparation of
reference |psi(v0=0.04)>. The conditional finite-log-price-window encoding
is kept fixed; missing probability outside the window is reported separately.

Generated results: fidelity curves, CNOT counts, folded/optimised angle CSVs,
selected Pauli strings, range summaries, and checkpoint OpenQASM files.
"""
from __future__ import annotations

# Fixed order copied from support_3_budget_16_trajectory in the preceding
# finite-path experiment. Embedding it makes this experiment reproducible from
# a fresh repository checkout without the previous experiment's CSV outputs.
CANDIDATE_WORDS = (
    "ZYIIIIII", "ZIYIIIII", "XYXIIIII", "IYZXIIII",
    "IZZYIIII", "ZZYIIIII", "IZXYIIII", "IYZZIIII",
    "IYIZIIII", "XIYIIIII", "IIYIZIII", "IXIYIIII",
    "IIZYXIII", "YIXIIIII", "IXYZIIII", "IIXYXIII",
)

import argparse
from dataclasses import dataclass, replace, asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v0_naive_tangent_generators import HestonFamily, Settings as FamilySettings, actions
from v0_finite_tangent_transport import matrix_of_generator
from finite_path_circuit_compilation import (
    CircuitAccumulator, pauli_rotation, simulate_gates, qasm_for_gates,
)

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


@dataclass(frozen=True)
class Settings:
    reference_v0: float = 0.04
    v0_min: float = 0.01
    v0_max: float = 0.40
    step: float = 0.002  # midpoint quadrature step; checkpoints at this resolution
    optimise_every: int = 5  # 0.01 in v0 for step=.002
    maxiter: int = 240
    threshold: float = 0.999
    qubits: int = 8
    n_fft: int = 16384
    cutoff: float = 0.7
    support: int = 3
    budget: int = 16
    method: str = 'trajectory'
    source: str = ''  # Optional: validate the frozen words against prior generator_sets.json
    output: str = 'heston_coninuity/results/single_pass_transport_circuit'
    export_targets: tuple[float,...] = (0.01,0.06,0.08,0.10,0.12,0.16,0.20,0.30,0.40)


def write_csv(path, rows):
    if not rows:
        return
    with open(path,'w',newline='') as handle:
        w=csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fidelity(a,b):
    return float(np.clip(np.dot(a,b)**2,0,1))


def select_words(s: Settings):
    key=f'support_{s.support}_budget_{s.budget}_{s.method}'
    if key!='support_3_budget_16_trajectory':
        raise ValueError('This is a single-candidate experiment: support 3, budget 16, trajectory')
    words=list(CANDIDATE_WORDS)
    if s.source:
        with open(s.source) as f:
            selections=json.load(f)
        if selections[key]!=words:
            raise AssertionError('Embedded candidate differs from supplied generator_sets.json')
    if len(words)!=s.budget or any(len(w)!=s.qubits for w in words):
        raise ValueError('Candidate does not match budget and qubit count')
    if any((sum(ch=='Y' for ch in w)%2)!=1 for w in words):
        raise ValueError('Non-real generator in selected candidate')
    return key,words


class PassCircuit:
    """Same fixed Pauli strings for every endpoint; only angles change."""
    def __init__(self, psi0, words):
        self.psi0=np.asarray(psi0,dtype=float)
        self.words=words
        self.K=[matrix_of_generator(w) for w in words]
        self.n=len(words)

    def propagate(self, theta, gradient_target=None):
        """Evaluate F and its analytic reverse-mode gradient if requested."""
        previous=self.psi0
        states=[]
        for a,K in zip(theta,self.K):
            previous=np.cos(a)*previous+np.sin(a)*(K @ previous)
            if gradient_target is not None:
                states.append(previous)
        if gradient_target is None:
            return previous
        overlap=float(gradient_target @ previous)
        F=min(1.0,max(0.0,overlap**2))
        back=gradient_target.copy()
        grad=np.empty(self.n,dtype=float)
        for j in range(self.n-1,-1,-1):
            derivative_action=self.K[j] @ states[j]
            grad[j]=-2*overlap*float(back @ derivative_action)
            angle=theta[j]
            back=np.cos(angle)*back-np.sin(angle)*(self.K[j] @ back)
        return 1.0-F,grad

    def endpoint_fit(self, target, folded, previous_solution, maxiter):
        starts=[np.asarray(folded,dtype=float)]
        if previous_solution is not None:
            starts.append(np.asarray(previous_solution,dtype=float))
        results=[]
        for initial in starts:
            res=minimize(lambda angles:self.propagate(angles,gradient_target=target),
                         initial, method='L-BFGS-B', jac=True,
                         options={'maxiter':maxiter,'ftol':2e-13,'gtol':2e-10})
            results.append(res)
        best=min(results,key=lambda x:x.fun)
        # no stochastic restarts, so this is not a global guarantee
        return np.asarray(best.x),float(1-best.fun),int(best.nit),bool(best.success)


def operator_only_last_qubits_upper_bound(ref,target,words):
    """Rigorous Uhlmann-type ceiling if all words act only on first m qubits.

    Any unitary supported only on the first m qubits preserves the reduced
    state on the last (n-m) qubits. Maximal possible pure-state fidelity under
    ALL unitaries on the first m qubits is nuclear_norm(B A^T)^2, where A,B
    reshape into (2**m, 2**(n-m)). Return None for a pool reaching all qubits.
    """
    m=max((i+1 for w in words for i,ch in enumerate(w) if ch!='I'),default=len(words[0]))
    n=len(words[0])
    if m>=n:
        return None,m
    a=ref.reshape(1<<m,1<<(n-m))
    b=target.reshape(1<<m,1<<(n-m))
    sv=np.linalg.svd(b @ a.T,compute_uv=False)
    return float(min(1,(sv.sum())**2)),m


def compile_pass(words,theta,layout):
    circuit=CircuitAccumulator(len(words[0]))
    for word,angle in zip(words,theta):
        circuit.extend(pauli_rotation(word,float(angle),layout))
    return circuit


def make_path(reference,end,step):
    n=int(round(abs(end-reference)/step))
    return np.linspace(reference,end,n+1)


def first_failure(rows,threshold):
    for row in rows[1:]:
        if row['folded_fidelity']<threshold:
            return float(row['v0'])
    return None


def fig_outputs(rows,out,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted(rows,key=lambda r:r['v0'])
    x=[r['v0'] for r in rows]
    fig,axs=plt.subplots(1,2,figsize=(13.8,5),constrained_layout=True)
    axs[0].plot(x,[r['folded_fidelity'] for r in rows],label='Folded tangent angles',lw=1.8)
    o=[r for r in rows if r['optimized_fidelity'] is not None]
    axs[0].scatter([r['v0'] for r in o],[r['optimized_fidelity'] for r in o],s=13,
                   label='Endpoint optimised (same circuit)',zorder=3)
    axs[0].plot(x,[r['identity_fidelity'] for r in rows],ls='--',label='Unchanged reference')
    axs[0].plot(x,[r['support_upper_bound'] for r in rows],ls=':',label='Any unitary on active qubits (ceiling)')
    axs[0].axhline(s.threshold,color='#bf4040',ls='--',alpha=.9,label=f'F={s.threshold}')
    axs[0].set(xlabel='Terminal Heston initial variance v₀',ylabel='Endpoint state fidelity',ylim=(0,1.005),title='One fixed 16-generator single-pass circuit')
    axs[0].legend(fontsize=8)
    axs[1].plot(x,[1-r['folded_fidelity'] for r in rows],label='Folded infidelity')
    axs[1].scatter([r['v0'] for r in o],[max(1-r['optimized_fidelity'],1e-14) for r in o],
                   s=13,label='Optimised infidelity')
    axs[1].plot(x,[1-r['support_upper_bound'] for r in rows],ls=':',label='Support-limited lower bound')
    axs[1].axhline(1-s.threshold,color='#bf4040',ls='--')
    axs[1].set(xlabel='v₀',ylabel='Infidelity 1-F',yscale='log',title='Error on logarithmic scale')
    axs[1].legend(fontsize=8);axs[1].grid(alpha=.25)
    for ax in axs:ax.axvline(s.reference_v0,color='grey',ls=':',alpha=.75);ax.grid(alpha=.2)
    fig.savefig(out/'one_pass_fidelity.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12.4,4.5),constrained_layout=True)
    for ax,name in ((axs[0],'cx'),(axs[1],'depth')):
        for layout,style in [('all_to_all','--'),('linear','-')]:
            ax.plot(x,[r[f'{layout}_{name}'] for r in rows],style,label=f'{layout} connectivity')
        ax.set(xlabel='v₀',ylabel=('CNOT count' if name=='cx' else 'Gate depth'))
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Single-pass cost: constant structure, continuously varying angles')
    fig.savefig(out/'one_pass_gate_cost.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12.5,4.5),constrained_layout=True)
    for name in ('tangent_leakage','cutoff_lost_mass'):
        ax=axs[0] if name=='tangent_leakage' else axs[1]
        ax.plot(x,[r[name] for r in rows])
        ax.set(xlabel='v₀',ylabel=('Relative squared tangent leakage' if name=='tangent_leakage' else 'Probability outside encoding window'))
        ax.grid(alpha=.2)
    fig.suptitle('Why fidelity changes: generator fit and finite encoding cutoff')
    fig.savefig(out/'tangent_leakage_and_cutoff.png',dpi=170);plt.close(fig)


def experiment(s):
    if not (0<s.v0_min<s.reference_v0<s.v0_max):
        raise ValueError('Need positive v0_min < reference < v0_max')
    if s.step<=0 or s.optimise_every<1 or s.threshold<=0 or s.threshold>=1:
        raise ValueError('Invalid integration or threshold settings')
    out=Path(s.output);(out/'qasm').mkdir(parents=True,exist_ok=True)
    family=HestonFamily(FamilySettings(qubits=s.qubits,n_fft=s.n_fft,cutoff=s.cutoff,
                                       reference_v0=s.reference_v0))
    key,words=select_words(s)
    ref=family.psi(s.reference_v0)
    circuit=PassCircuit(ref,words)
    ceiling,m=operator_only_last_qubits_upper_bound(ref,ref,words)
    (out/'candidate.json').write_text(json.dumps({'candidate_key':key,'ordered_pauli_words':words,
      'active_qubits_zero_based':sorted(set(i for w in words for i,ch in enumerate(w) if ch!='I')),
      'fixed_gate_structure':True,'angle_convention':'exp(-i theta_k P_k)',
      'remark':'Optimised endpoint angles use exact target amplitudes; oracle benchmark, not learned prediction.'},indent=2)+'\n')
    rows=[];angles_rows=[];summary=[]
    for branch,end in [('decreasing',s.v0_min),('increasing',s.v0_max)]:
        path=make_path(s.reference_v0,end,s.step)
        theta=np.zeros(len(words))
        previous_opt=None
        branch_rows=[]
        for k,v in enumerate(path):
            v=float(v)
            exact,tangent,kept,lost,edge,negative=family.probabilities_and_derivative(v)
            if k:
                mid=(float(path[k-1])+v)/2
                mid_state,mid_t,*_=family.probabilities_and_derivative(mid)
                A=actions(mid_state,words)
                coefs=np.linalg.lstsq(A,mid_t,rcond=1e-10)[0]
                theta=theta+coefs*(v-float(path[k-1]))
            else:
                coefs=np.zeros(len(words))
            folded_state=circuit.propagate(theta)
            F_fold=fidelity(folded_state,exact)
            F_ref=fidelity(ref,exact)
            ceiling,_=operator_only_last_qubits_upper_bound(ref,exact,words)
            if k%s.optimise_every==0 or (branch=="increasing" and v<=0.15) or k==len(path)-1:
                opt_angles,opt_F,nit,success=circuit.endpoint_fit(exact,theta,previous_opt,s.maxiter)
                previous_opt=opt_angles.copy()
            else:
                opt_angles,opt_F,nit,success=None,None,None,None
            columns=actions(exact,words)
            cc=np.linalg.lstsq(columns,tangent,rcond=1e-10)[0]
            resid=tangent-columns@cc
            tangent_leak=float(np.dot(resid,resid)/np.dot(tangent,tangent))
            counts={}
            for layout in ('all_to_all','linear'):
                gates=compile_pass(words,theta,layout)
                metrics=gates.metrics()
                counts[f'{layout}_cx']=metrics['cx']
                counts[f'{layout}_depth']=metrics['depth']
                counts[f'{layout}_rz']=metrics['rz']
                counts[f'{layout}_total_gates']=metrics['gates']
                if any(abs(v-ex)<s.step*.51 for ex in s.export_targets) and k:
                    tag=f'{branch}_v0_{v:.3f}'.replace('.','p')
                    file=out/'qasm'/f'folded_{layout}_{tag}.qasm'
                    gate_seq=list(gates.active())
                    check=simulate_gates(ref,gate_seq)
                    err=np.linalg.norm(check-folded_state)
                    if err>5e-8:
                        raise AssertionError(f'Gate-level state mismatch: {err} ({file})')
                    file.write_text(qasm_for_gates(gate_seq,s.qubits,
                        metadata=f'Fixed candidate {key}; target v0={v}; folded-angle single pass'))
                    if opt_angles is not None:
                        gates_opt=compile_pass(words,opt_angles,layout)
                        check_opt=simulate_gates(ref,list(gates_opt.active()))
                        if np.linalg.norm(check_opt-circuit.propagate(opt_angles))>5e-8:
                            raise AssertionError('Optimised gate-level circuit disagrees with Pauli evolution')
                        (out/'qasm'/f'optimized_{layout}_{tag}.qasm').write_text(
                           qasm_for_gates(list(gates_opt.active()),s.qubits,
                             metadata=f'Fixed candidate {key}; target v0={v}; oracle endpoint-optimised angles'))
            row={'branch':branch,'step_index':k,'v0':v,'distance':abs(v-s.reference_v0),
                 'folded_fidelity':F_fold,'folded_infidelity':1-F_fold,
                 'optimized_fidelity':opt_F,'optimized_iterations':nit,
                 'optimized_solver_success':success,'identity_fidelity':F_ref,
                 'support_upper_bound':ceiling,'tangent_leakage':tangent_leak,
                 'cutoff_kept_mass':kept,'cutoff_lost_mass':lost,
                 'fft_edge_mass':edge,'fft_negative_mass_clipped':negative,**counts}
            branch_rows.append(row)
            for j,w in enumerate(words):
                angles_rows.append({'branch':branch,'v0':v,'pauli_string':w,'order':j+1,
                                    'folded_angle':float(theta[j]),'midpoint_coefficient':float(coefs[j]),
                                    'optimized_angle':(float(opt_angles[j]) if opt_angles is not None else None)})
            if k%25==0 or k==len(path)-1:
                print(f'{branch:10} v0={v:.4f} folded F={F_fold:.7f} '
                      f'optimised F={opt_F if opt_F is not None else "..."} '
                      f'linear-CX={counts["linear_cx"]} lostmass={lost:.4f}',flush=True)
        rows.extend(branch_rows)
        first_fold=next((r['v0'] for r in branch_rows[1:] if r['folded_fidelity']<s.threshold),None)
        optimized_rows=[r for r in branch_rows if r['optimized_fidelity'] is not None]
        first_opt=next((r['v0'] for r in optimized_rows[1:] if r['optimized_fidelity']<s.threshold),None)
        summary.append({'branch':branch,'target_range_endpoint':end,'fidelity_threshold':s.threshold,
                        'first_folded_failure':first_fold,'first_optimized_sample_failure':first_opt,
                        'last_folded_fidelity':branch_rows[-1]['folded_fidelity'],
                        'last_optimized_fidelity':branch_rows[-1]['optimized_fidelity'],
                        'folded_worst_fidelity':min(r['folded_fidelity'] for r in branch_rows),
                        'optimized_worst_sample_fidelity':min(r['optimized_fidelity'] for r in optimized_rows),
                        'nearest_neighbour_cx_at_endpoint':branch_rows[-1]['linear_cx'],
                        'all_to_all_cx_at_endpoint':branch_rows[-1]['all_to_all_cx']})
    write_csv(out/'single_pass_sweep.csv',rows)
    write_csv(out/'angle_history.csv',angles_rows)
    write_csv(out/'boundary_summary.csv',summary)
    (out/'settings.json').write_text(json.dumps(asdict(s),indent=2)+'\n')
    fig_outputs(rows,out,s)
    print('\nSummary:')
    for r in summary:
        print(r)
    print('Output:',out.resolve())
    return rows,summary


def smoke_test():
    rng=np.random.default_rng(82)
    words=['ZYII','IXYI','XXYI']
    state=rng.normal(size=16);state/=np.linalg.norm(state)
    obj=PassCircuit(state,words)
    angles=np.array([.13,-.24,.05]);target=rng.normal(size=16);target/=np.linalg.norm(target)
    f,g=obj.propagate(angles,target)
    for j in range(3):
        h=1e-6
        pos=angles.copy();pos[j]+=h
        neg=angles.copy();neg[j]-=h
        numeric=(obj.propagate(pos,target)[0]-obj.propagate(neg,target)[0])/(2*h)
        assert abs(numeric-g[j])<3e-8,(j,numeric,g[j])
    for connectivity in ('all_to_all','linear'):
        compiled=compile_pass(words,angles,connectivity)
        real=simulate_gates(state,list(compiled.active()))
        expected=obj.propagate(angles)
        assert np.linalg.norm(real-expected)<1e-12,(connectivity,np.linalg.norm(real-expected))
    print('Smoke tests passed: analytic angle gradients and gate-level compilation')


def main():
    default=Settings();p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name,kwargs in [('v0-min',{'type':float}),('v0-max',{'type':float}),
                        ('step',{'type':float}),('optimise-every',{'type':int}),
                        ('maxiter',{'type':int}),('threshold',{'type':float}),
                        ('source',{'type':str}),('output',{'type':str})]:
        p.add_argument('--'+name,default=getattr(default,name.replace('-','_')),**kwargs)
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    smoke_test()
    if a.smoke_only:
        return
    config=replace(default,v0_min=a.v0_min,v0_max=a.v0_max,step=a.step,
                   optimise_every=a.optimise_every,maxiter=a.maxiter,
                   threshold=a.threshold,source=a.source,output=a.output)
    if threadpool_limits is None:
        experiment(config)
    else:
        with threadpool_limits(limits=1,user_api='blas'):
            experiment(config)


if __name__=='__main__':
    main()