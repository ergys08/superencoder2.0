#!/usr/bin/env python3
"""Chronological, five-parameter single-pass Heston transport and reference resets.

Run from repo root:
  python heston_2013_transport_references.py
  python heston_2013_transport_references.py --data /path/to/2013.csv --budget 16

FIRST observation must be the first reference. Every next trading day:
  (1) attempt ONE fixed, reference-specific, K-Pauli-rotation circuit using
      exactly the currently active reference state;
  (2) determine its K angles with multi-start target-state fidelity fitting;
  (3) if its directly simulated fidelity is below --minimum-fidelity, create
      a new exact reference on that date and rebuild its tangent-selected set;
  (4) track full circuit-level fidelity for EVERY date, not only selected days.

No future observations select references/generators. Endpoint optimisation has
access to the actual current-day Heston target state, making the method an
ORACLE EXPRESSIVITY test, NOT a ready-to-deploy cheap neural angle predictor.
Local multistart optimisation failure IS NOT proof of intrinsic nonexpressivity.
Each reference reset is direct state preparation; it does not stack transports.
We separately log truncation loss; resets CANNOT fix encoding-window loss.
Gate counts exclude reference preparation and are not runtime/noise estimates.
"""
from __future__ import annotations
import argparse,csv,json,sys,time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
HERE=ROOT/'heston_coninuity'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from heston_calibrations import read_panel,PARAMETERS
from five_parameter_tangent_generators import (Settings, vector_to_z,jacobian,make_encoder,
    library,select_shared,diagnostics)
from v0_naive_tangent_generators import actions
from single_pass_transport_circuit import PassCircuit,compile_pass
from finite_path_circuit_compilation import simulate_gates,qasm_for_gates

try:from threadpoolctl import threadpool_limits
except ImportError:threadpool_limits=None


def set_seed_starts(psi0,target,circuit,previous,extra_starts=0):
    """Physics-motivated starts; all target dependent (therefore oracle)."""
    n=len(circuit.words)
    linear=np.linalg.lstsq(actions(psi0,circuit.words),target-psi0,rcond=1e-9)[0]
    seeds=[np.zeros(n),linear,0.5*linear]
    if previous is not None:
        seeds.insert(0,np.asarray(previous).copy())
        seeds.append((np.asarray(previous)+linear)/2)
    if extra_starts:
        rng=np.random.default_rng(12345)
        for i in range(extra_starts):seeds.append(linear+rng.normal(0,.12,size=n))
    return seeds


def optimise_pass(circuit,target,previous,maxiter,threshold,restarts=0):
    """Best of deterministic, target-dependent starts; exit early on success.

    No successful fit is allowed to claim fidelity above its independently
    recalculated circuit fidelity. The returned result is NOT a global optimum.
    """
    seeds=set_seed_starts(circuit.psi0,target,circuit,previous,restarts)
    best=(float('-inf'),np.zeros(len(circuit.words)),0)
    evaluations=0
    for seed in seeds:
        res=minimize(lambda x:circuit.propagate(x,gradient_target=target),seed,
          jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-13,'gtol':1e-9})
        evaluations+=int(res.nfev)
        predicted=circuit.propagate(res.x)
        f=float(np.clip(np.dot(predicted,target)**2,0,1))
        if f>best[0]:best=(f,res.x.copy(),int(res.nit))
        if f >= threshold+1e-10:break
    return best,evaluations


def make_reference(enc,panel_row,words_pool,budget,fd_step):
    z=vector_to_z(panel_row)
    psi,J,kept,fd=jacobian(enc,z,fd_step)
    words=select_shared(psi,J,words_pool,budget)
    circuit=PassCircuit(psi,words)
    diag=diagnostics(psi,J,words)
    # Compilation costs from a generic, nonzero angle vector: specific angles
    # may be simplified further at each day. Metrics exclude reference loader.
    metrics={}
    for layout in ('linear','all_to_all'):
        compiler=compile_pass(words,np.ones(len(words))*0.113,layout)
        metrics[layout]=compiler.metrics()
    return z,psi,words,circuit,diag,fd,metrics


def run(args):
    if not (0<args.minimum_fidelity<=1):raise ValueError('Invalid fidelity requirement')
    panel=read_panel('2013',args.data,ROOT/'heston_calibration_results'/'data')
    if args.n_dates>0:panel=panel.iloc[:args.n_dates].copy()
    if panel.degenerate.any():
        raise ValueError('Flagged calibrations encountered; define how to treat them')
    enc=make_encoder(Settings(qubits=args.qubits,fft_size=args.n_fft,
        fft_halfwidth=args.fft_halfwidth,cutoff=args.cutoff,
        maturity=args.maturity,drift=args.drift,
        budget=args.budget,support=args.support,fd_step=args.fd_step))
    pool=library(args.qubits,args.support)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    all_states=[];all_mass=[];all_edge=[]
    for i,row in panel.iterrows():
        psi,m,negative,edge=enc.state(row)
        all_states.append(psi);all_mass.append(m);all_edge.append(edge)
    all_states=np.stack(all_states)
    # Fair chronological zero-generator control: keep yesterday's anchor until
    # raw reference-target overlap misses the SAME fidelity requirement.
    identity_reference_indices=[0]
    identity_anchor=0
    for j in range(1,len(all_states)):
        if float((all_states[j] @ all_states[identity_anchor])**2)<args.minimum_fidelity:
            identity_anchor=j
            identity_reference_indices.append(j)
    old_reference_date=None
    references=[];rows=[];previous_theta=None
    for i,(_,entry) in enumerate(panel.iterrows()):
        target=all_states[i]
        true_mass=all_mass[i]
        if i>0:
            prev_identity=float(np.dot(target,all_states[i-1])**2)
        else: prev_identity=1.0
        active_index=references[-1]['index'] if references else 0
        active_psi=all_states[active_index]
        identity_from_reference=float(np.dot(target,active_psi)**2)
        reinitialised=not references
        attempts=0;nfev=0;fidelity_before_reset=1.0
        if references:
            (fit,theta,iterations),nfev=optimise_pass(circuit,target,previous_theta,
                        args.maxiter,args.minimum_fidelity,args.extra_restarts)
            attempts=1
            fidelity_before_reset=fit
            if fit < args.minimum_fidelity-1e-12:
                reinitialised=True
        if reinitialised:
            ref_z,ref_psi,words,circuit,diag,fd,metric=make_reference(enc,entry,
                pool,args.budget,args.fd_step)
            ref_idx=len(references)
            references.append({'ref_number':ref_idx,'index':i,'date':str(entry.date.date()),
                  'heston_params':{k:float(entry[k]) for k in PARAMETERS},
                  'chosen_words':words,'local_tangent':diag,'fd_check':fd,
                  'generic_circuit_cost':metric})
            previous_theta=None
            theta=np.zeros(len(words))
            produced=ref_psi
            measured_fidelity=float(np.clip(np.dot(target,produced)**2,0,1))
            cost={'cx':0,'gates':0,'depth':0}
            independent_error=0.0
            decision='new_reference'
        else:
            previous_theta=theta
            produced=circuit.propagate(theta)
            measured_fidelity=float(np.clip(np.dot(target,produced)**2,0,1))
            logical=compile_pass(words,theta,'linear')
            cost=logical.metrics()
            if args.verify_all_compiled:
                physical=simulate_gates(ref_psi,list(logical.active()))
                independent_error=float(np.linalg.norm(physical-produced))
                if independent_error > 1e-7: raise AssertionError(f'QASM compiler disagreement {entry.date}: {independent_error}')
                measured_fidelity=float(np.clip(np.abs(np.vdot(target,physical))**2,0,1))
            else:independent_error=None
            decision='transport'
        if measured_fidelity<args.minimum_fidelity-1e-9:
            raise AssertionError(f'Below threshold after decision: {entry.date}, fidelity {measured_fidelity}')
        row={'date':str(entry.date.date()),'date_index':i,'decision':decision,
            'reference_number':len(references)-1,'reference_date':references[-1]['date'],
            'reference_age_days':i-references[-1]['index'],
            'identity_fidelity_from_previous_day':prev_identity,
            'identity_fidelity_from_reference':identity_from_reference,
            'fidelity_of_attempt_before_reset':fidelity_before_reset,
            'actual_output_fidelity':measured_fidelity,
            'actual_output_infidelity':1-measured_fidelity,
            'lost_mass':1-true_mass,'full_window_overlap_fidelity':true_mass*measured_fidelity,
            'transport_cx':cost['cx'],'transport_depth':cost['depth'],
            'transport_total_gates':cost['gates'],
            'gate_simulation_error_norm':independent_error,
            'optimizer_evaluations':nfev,
            'angle_vector_json':json.dumps(theta.tolist(),separators=(',',':')),
            'v0':float(entry.v0),'kappa':float(entry.kappa),'theta':float(entry.theta),
            'xi':float(entry.xi),'rho':float(entry.rho)}
        rows.append(row)
        if args.progress_every>0 and (i%args.progress_every==0 or reinitialised or i==len(panel)-1):
            print(f'{i+1}/{len(panel)} {row["date"]} {decision:13s} ref={row["reference_number"]:3d} '
                f'F={measured_fidelity:.7f} (attempt={fidelity_before_reset:.7f}) lost={row["lost_mass"]:.3%}',flush=True)
    with (out/'daily_replay.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    regions=[]
    for idx,ref in enumerate(references):
        entries=[r for r in rows if r['reference_number']==idx]
        regions.append({'reference':idx,'start_date':ref['date'],'end_date':entries[-1]['date'],
            'n_observations':len(entries),'n_transports':sum(r['decision']=='transport' for r in entries),
            'worst_fidelity':min(r['actual_output_fidelity'] for r in entries),
            'max_reference_age_days':max(r['reference_age_days'] for r in entries)})
    transports=[r for r in rows if r['decision']=='transport']
    results={'data_source':'https://github.com/oskarimikkila/Empirical-Deep-Hedging/blob/master/data/heston_params.csv',
        'data_dates':[rows[0]['date'],rows[-1]['date']],
        'n_dates':len(rows),'n_references':len(references),'identity_only_chronological_references':len(identity_reference_indices),
        'identity_reference_indices':identity_reference_indices,'n_successful_transports':len(transports),
        'fraction_days_using_transport':len(transports)/len(rows),
        'minimum_achieved_conditional_fidelity':min(r['actual_output_fidelity'] for r in rows),
        'minimum_achieved_full_window_overlap_fidelity':min(r['full_window_overlap_fidelity'] for r in rows),
        'max_lost_mass':max(r['lost_mass'] for r in rows),
        'median_lost_mass':float(np.median([r['lost_mass'] for r in rows])),
        'median_transport_fidelity':float(np.median([r['actual_output_fidelity'] for r in transports])) if transports else None,
        'mean_reference_length_observations':float(np.mean([r['n_observations'] for r in regions])),
        'median_reference_length_observations':float(np.median([r['n_observations'] for r in regions])),
        'longest_reference_length_observations':max(r['n_observations'] for r in regions),
        'total_transport_cx_linear':sum(r['transport_cx'] for r in rows),
        'mean_successful_transport_cx_linear':float(np.mean([r['transport_cx'] for r in transports])) if transports else None,
        'largest_compiler_validation_error':max(r['gate_simulation_error_norm'] or 0 for r in rows),
        'minimum_fidelity_threshold':args.minimum_fidelity,
        'interpretation':'Local optimisation failure triggers conservative reset; does not certify ansatz inexpressivity. '
                         'Every outcome is validated; resets do not recover mass outside encoding window. '
                         'Angle fitting uses current exact target and is computationally expensive; neural predictor not yet supplied.',
        'settings':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}}
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'reference_regions.json').write_text(json.dumps(regions,indent=2)+'\n')
    (out/'reference_generators.json').write_text(json.dumps(references,indent=2)+'\n')
    plot(rows,regions,out,args.minimum_fidelity)
    print('\n'+json.dumps({k:v for k,v in results.items() if k!='settings'},indent=2))
    return results


def plot(rows,regions,out,threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=np.arange(len(rows));dates=[r['date'] for r in rows]
    fig,ax=plt.subplots(3,1,figsize=(13.5,9),sharex=True,constrained_layout=True)
    for p in ax: p.grid(alpha=.22)
    f=np.array([r['actual_output_fidelity'] for r in rows])
    a=np.array([r['fidelity_of_attempt_before_reset'] for r in rows])
    ax[0].plot(n,f,label='Actual output fidelity (including resets)',lw=1.2)
    ax[0].scatter(n,a,s=7,alpha=.35,label='Pre-reset attempted fidelity')
    ax[0].axhline(threshold,color='red',ls='--',label=f'Requirement {threshold}')
    ax[0].set(ylabel='Conditional fidelity',title='2013 chronological 5-parameter single-pass transport replay')
    ax[0].legend(fontsize=8,loc='lower left')
    refs=[r['date_index'] for r in rows if r['decision']=='new_reference']
    ax[1].step(n,[r['reference_number'] for r in rows],where='post',label='Reference ID')
    for ix in refs:ax[1].axvline(ix,color='grey',alpha=.10)
    ax[1].set(ylabel='Reference number')
    ax[2].plot(n,[r['lost_mass'] for r in rows],label='Window mass loss')
    ax[2].set(ylabel='Missing probability',xlabel='Trading observation index')
    labels=[0,*list(range(19,len(rows),20)),len(rows)-1]
    ax[2].set_xticks(labels,[dates[i] for i in labels],rotation=35,ha='right',fontsize=8)
    fig.savefig(out/'daily_reference_replay.png',dpi=160);plt.close(fig)
    fig,bx=plt.subplots(figsize=(10.5,4.2),constrained_layout=True)
    bx.hist([r['n_observations'] for r in regions],bins=np.arange(.5,1.5+max(r['n_observations'] for r in regions)),align='mid')
    bx.set(xlabel='Number of consecutive trading observations per reference',ylabel='Reference count',
        title='Distribution of sequential transport-region lifetimes')
    fig.savefig(out/'reference_lifetimes.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,help='Offline copy of 2013 published parameter panel')
    p.add_argument('--n-dates',type=int,default=0,help='0 means full 250-day panel')
    p.add_argument('--minimum-fidelity',type=float,default=.999)
    p.add_argument('--support',type=int,default=3)
    p.add_argument('--budget',type=int,default=16)
    p.add_argument('--maxiter',type=int,default=175)
    p.add_argument('--extra-restarts',type=int,default=0)
    p.add_argument('--qubits',type=int,default=8)
    p.add_argument('--n-fft',type=int,default=16384)
    p.add_argument('--fft-halfwidth',type=float,default=3.5)
    p.add_argument('--cutoff',type=float,default=.7)
    p.add_argument('--maturity',type=float,default=1.0)
    p.add_argument('--drift',type=float,default=.05)
    p.add_argument('--fd-step',type=float,default=.0005)
    p.add_argument('--progress-every',type=int,default=25)
    p.add_argument('--verify-all-compiled',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--output',default='heston_calibration_results/five_parameter_reference_replay')
    args=p.parse_args()
    if threadpool_limits:
        with threadpool_limits(limits=1,user_api='blas'):
            run(args)
    else:run(args)

if __name__=='__main__':main()