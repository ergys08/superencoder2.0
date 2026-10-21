#!/usr/bin/env python3
"""Auditable boundaries for ONE fixed single-pass Heston v0 transport circuit.

Re-run previous sweep first (same fixed bins/FFT/candidate):
    python heston_coninuity/single_pass_transport_circuit.py
Then, from repository root:
    python heston_coninuity/single_pass_circuit_boundaries.py

Definitions, without conflating them:
  - Conditional fidelity F=|<psi_cond(v)|U(theta)psi_cond(ref)>|^2.
    The *intrinsic ansatz* expressivity criterion is max_theta F >= F_min.
    Multi-start endpoint-optimised values in the upstream CSV are only LOWER
    BOUNDS on max_theta F, not certified global optima. Failed optimisations
    are inconclusive. Folded angle results are constructive fidelity witnesses.
  - Encoding mass M=P(log S_T inside fixed window). The encoding criterion is
    1-M <= epsilon. It is separate from the circuit.
  - A full-state overlap proxy is M*F (with an orthogonal outside-window
    space and consistent bins). This is *not* total variation distance or a
    payoff-error guarantee.
  - Break-even C_ref + C_transport <= C_fresh(v) uses the SAME CNOT metric and
    connectivity; fixed-angle single-pass C_transport does not depend on v.
    Costs ignore classical preprocessing, execution noise and target-observable
    error. User-specified direct loading counts are explicit comparators, NOT
    proven optima. A cost boundary exists only if comparator costs cross the
    fixed total. With constant direct-loader counts, there is no v boundary.

Crossing points are *sample brackets*, not guaranteed unique boundaries or
continuous-function roots. Branches are analysed from the reference outward,
without assuming monotonicity. Every failed local optimisation is flagged
as non-certifying for an expressivity limit.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'results/single_pass_transport_circuit/single_pass_sweep.csv'
DEFAULT_OUT = ROOT / 'results/single_pass_circuit_boundaries'


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def as_float(v):
    return float(v) if v not in ('', None, 'None', 'nan') else None


def parse_sweep(path):
    rows=[]
    for item in read_csv(path):
        row={k:(as_float(item[k]) if k in (
            'v0','distance','folded_fidelity','optimized_fidelity','support_upper_bound',
            'cutoff_kept_mass','cutoff_lost_mass','all_to_all_cx','linear_cx',
            'linear_depth','all_to_all_depth','fft_edge_mass') else item[k])
             for k in item}
        rows.append(row)
    if not rows: raise ValueError('Empty input sweep')
    for r in rows:
        for x in ('v0','folded_fidelity','cutoff_kept_mass','cutoff_lost_mass','support_upper_bound'):
            if r[x] is None or not np.isfinite(r[x]):raise ValueError(f'Bad {x} at {r.get("v0")}')
        if abs(r['cutoff_kept_mass']+r['cutoff_lost_mass']-1)>2e-7:
            raise ValueError('Mass accounting inconsistent')
        if not 0<=r['folded_fidelity']<=1+1e-8:raise ValueError('Invalid fidelity')
        if r['folded_fidelity']>r['support_upper_bound']+2e-7:
            raise ValueError('Folded fidelity violates the physical support ceiling')
        if r['optimized_fidelity'] is not None:
            if r['optimized_fidelity']>r['support_upper_bound']+2e-7:
                raise ValueError('Optimised fidelity violates the physical support ceiling')
            if r['optimized_fidelity']+2e-7<r['folded_fidelity']:
                raise ValueError('Endpoint optimisation worse than folded witness')
    return rows


def branches_from_ref(rows, ref):
    out={}
    for branch in ('decreasing','increasing'):
        data=sorted((r for r in rows if r['branch']==branch),key=lambda r:abs(r['v0']-ref))
        if not data or abs(data[0]['v0']-ref)>1e-6:raise ValueError(f'No reference in {branch}')
        if len(data)!=len({round(r['v0'],9) for r in data}): raise ValueError('Duplicate v0')
        out[branch]=data
    return out


def crossings(rows, predicate):
    """First and all sampled PASS->FAIL transitions and recoveries outward."""
    prev=rows[0]; prev_ok=bool(predicate(prev)); turns=[]
    for r in rows[1:]:
        ok=bool(predicate(r))
        if ok!=prev_ok:
            turns.append({'type':'first_failure' if prev_ok else 'recovery',
                          'previous_v0':prev['v0'],'next_v0':r['v0'],
                          'bracket_low':min(prev['v0'],r['v0']),
                          'bracket_high':max(prev['v0'],r['v0'])})
        prev,prev_ok=r,ok
    failures=[t for t in turns if t['type']=='first_failure']
    return {'first_failure_bracket':failures[0] if failures else None,
            'all_transitions':turns,
            'ever_failed_on_grid':any(not predicate(r) for r in rows),
            'sample_count':len(rows)}


def numeric_metric(row, name):
    m=row['cutoff_kept_mass']; f=row['folded_fidelity']
    if name=='mass_loss':return 1-m
    if name=='folded_fidelity':return f
    if name=='full_folded':return m*f
    if name=='support_ceiling':return row['support_upper_bound']
    if name=='optimized_fidelity':return row['optimized_fidelity']
    raise ValueError(name)


def default_constant_direct(loader, layout):
    # Auditable, manually specified comparison to the prior exact 8-qubit
    # uniformly-controlled-Ry loader. NOT a theoretical/optimised lower bound.
    return loader[layout]


def cost_table(rows, ref, baseline, reference_cost, layout):
    key=layout+'_cx'
    nonref=[r for r in rows if abs(r['v0']-ref)>1e-9]
    distinct={int(r[key]) for r in nonref}
    if len(distinct)!=1:
        raise ValueError(f'Nonconstant transport CX count: {sorted(distinct)}')
    k=distinct.pop()
    total=reference_cost+k
    out=[]
    for r in rows:
        target_cx=baseline(r['v0'])
        if target_cx is None: raise ValueError('Missing target baseline')
        out.append({'branch':r['branch'],'v0':r['v0'],
                    'reference_cx':reference_cost,
                    'transport_cx':0 if abs(r['v0']-ref)<1e-9 else k,
                    'transport_total_cx':reference_cost+(0 if abs(r['v0']-ref)<1e-9 else k),
                    'direct_cx':target_cx,
                    'transport_minus_direct_cx':reference_cost+(0 if abs(r['v0']-ref)<1e-9 else k)-target_cx,
                    'break_even':reference_cost+(0 if abs(r['v0']-ref)<1e-9 else k)<=target_cx})
    return out,{'reference_cx':reference_cost,'fixed_single_pass_cx':k,
                'total_nonreference_cx':total,'layout':layout}


def plot(rows, costs, out, fidelity_threshold, loss_threshold, full_threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    one={round(r['v0'],9):r for r in rows}
    data=[one[x] for x in sorted(one)]
    xs=[r['v0'] for r in data]
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    a=axs[0,0]
    a.plot(xs,[r['folded_fidelity'] for r in data],label='Folded-angle witness')
    a.scatter([r['v0'] for r in data if r['optimized_fidelity'] is not None],
              [r['optimized_fidelity'] for r in data if r['optimized_fidelity'] is not None],s=12,label='Local endpoint fit (witness)')
    a.plot(xs,[r['support_upper_bound'] for r in data],ls=':',label='Support-based upper bound')
    a.axhline(fidelity_threshold,ls='--',color='black',label='Fidelity requirement')
    a.set(xlabel='v₀',ylabel='Conditional state fidelity',title='Expressivity: witnesses versus necessary ceiling')
    a.legend(fontsize=8)
    a=axs[0,1]
    a.plot(xs,[r['cutoff_lost_mass'] for r in data],label='Discarded probability mass')
    a.axhline(loss_threshold,ls='--',color='black',label='Mass threshold')
    a.set(xlabel='v₀',ylabel='Outside-window probability',title='Encoding boundary (independent of transport)')
    a.legend(fontsize=8)
    a=axs[1,0]
    a.plot(xs,[r['cutoff_kept_mass']*r['folded_fidelity'] for r in data],label='M × folded fidelity')
    a.axhline(full_threshold,ls='--',color='black',label='Full-overlap requirement')
    a.set(xlabel='v₀',ylabel='Full-space overlap fidelity proxy',title='Combined error (orthogonal tail embedding)')
    a.legend(fontsize=8)
    a=axs[1,1]
    dedup={round(r['v0'],9):r for r in costs}; co=[dedup[x] for x in sorted(dedup)]
    a.plot([r['v0'] for r in co],[r['transport_total_cx'] for r in co],label='Reference + fixed single pass')
    a.plot([r['v0'] for r in co],[r['direct_cx'] for r in co],ls='--',label='Direct loader baseline')
    a.set(xlabel='v₀',ylabel='CNOT count',title='Gate-cost comparison, same connectivity')
    a.legend(fontsize=8)
    for a in axs.flat:a.grid(alpha=.25)
    fig.savefig(out/'boundaries.png',dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--reference',type=float,default=.04)
    p.add_argument('--fidelity-min',type=float,default=.999)
    p.add_argument('--max-lost-mass',type=float,default=.01)
    p.add_argument('--full-fidelity-min',type=float,default=.99)
    p.add_argument('--connectivity',choices=('linear','all_to_all'),default='linear')
    p.add_argument('--reference-cx',type=int,default=None,
                   help='Override reference preparation CNOT count for this connectivity')
    p.add_argument('--direct-cx',type=int,default=None,
                   help='Constant direct preparation comparator; omit to use explicit existing loader count')
    p.add_argument('--direct-cx-csv',type=Path,default=None,
                   help='Optional CSV with columns v0,direct_cx; interpolate numeric baselines')
    args=p.parse_args()
    if not (0<args.fidelity_min<=1 and 0<=args.max_lost_mass<1 and 0<args.full_fidelity_min<=1):
        p.error('Invalid criteria')
    rows=parse_sweep(args.source)
    branches=branches_from_ref(rows,args.reference)
    baseline_origin='Previous direct eight-qubit uniformly controlled Ry loader, unoptimised routing'
    known={'linear':1502,'all_to_all':254}
    base_ref=known[args.connectivity] if args.reference_cx is None else args.reference_cx
    if args.direct_cx_csv:
        b=read_csv(args.direct_cx_csv)
        values=sorted((float(x['v0']),float(x['direct_cx'])) for x in b)
        if len(values)<2 or len(set(x[0] for x in values))!=len(values):p.error('Invalid comparator CSV')
        def loader(v):
            if v<values[0][0]-1e-10 or v>values[-1][0]+1e-10:
                raise ValueError(f'Baseline not supplied for v0 {v}')
            return float(np.interp(v,[x[0] for x in values],[x[1] for x in values]))
        baseline_origin='User-supplied v0-dependent direct-preparation CNOT measurements (linear interpolation)'
    else:
        direct=known[args.connectivity] if args.direct_cx is None else args.direct_cx
        loader=lambda v:direct
        baseline_origin+='; constant '+str(direct)+' CX'
    costs,meta=cost_table(rows,args.reference,loader,base_ref,args.connectivity)
    cost_branches=branches_from_ref(costs,args.reference)
    results={'definitions':{
        'true_expressivity':'sup_theta F_cond(theta,v0) >= F_min; NOT certified by local optimisation',
        'folded':'Constructive fixed-angle fidelity witness',
        'optimized':'Local optimiser result is a constructive witness, NOT global maximum',
        'necessary_impossibility':'F_support_ceiling(v0) < F_min certifies fixed-support impossibility',
        'encoding':'1-M(v0) <= max_lost_mass independent of quantum circuit',
        'full_overlap':'M(v0)*F_cond(v0) with orthogonal out-of-window space',
        'economics':'C_ref + C_transport <= C_direct(v0); same connectivity and gate set',
        'crossing':'Grid brackets only, no monotonicity or unique continuous root assumed'},
        'thresholds':{'fidelity_min':args.fidelity_min,'max_lost_mass':args.max_lost_mass,
                      'full_fidelity_min':args.full_fidelity_min},
        'cost_comparator':{'source':baseline_origin,**meta},'branches':{}}
    enriched=[]
    for branch,data in branches.items():
        b={}
        predicates={
            'folded_fidelity_witness':lambda r:r['folded_fidelity']>=args.fidelity_min,
            'mass_criterion':lambda r:r['cutoff_lost_mass']<=args.max_lost_mass,
            'folded_full_overlap':lambda r:r['cutoff_kept_mass']*r['folded_fidelity']>=args.full_fidelity_min,
            'support_ceiling_necessary':lambda r:r['support_upper_bound']>=args.fidelity_min,
        }
        for key,fn in predicates.items():b[key]=crossings(data,fn)
        opt_data=[r for r in data if r['optimized_fidelity'] is not None]
        b['optimized_sample_witness']=crossings(opt_data,lambda r:r['optimized_fidelity']>=args.fidelity_min)
        b['optimized_sample_warning']='Failure of local fit NOT proof of expressivity failure; unsampled points unknown'
        nonref_cost=cost_branches[branch][1:]
        b['cost_comparison_nonreference_constant']=len({x['direct_cx'] for x in nonref_cost})==1
        b['cost_break_even']=(crossings(nonref_cost,lambda r:r['break_even'])
                              if nonref_cost else None)
        b['cost_boundary_interpretation']=(
           'No parameter-dependent crossover exists for this constant-cost direct loader and fixed single-pass circuit; the reference point has zero additional transport gates.'
           if b['cost_comparison_nonreference_constant'] else
           'Target-dependent comparator: crossing brackets describe sampled cost changes; no monotonicity assumed.')
        b['sampled_range']=[min(r['v0'] for r in data),max(r['v0'] for r in data)]
        results['branches'][branch]=b
        for r in data:
            enriched.append({'branch':branch,'v0':r['v0'],'folded_conditional_fidelity':r['folded_fidelity'],
                  'optimized_conditional_fidelity_if_sampled':r['optimized_fidelity'],
                  'support_fidelity_upper_bound':r['support_upper_bound'],
                  'kept_mass':r['cutoff_kept_mass'],'lost_mass':r['cutoff_lost_mass'],
                  'folded_full_overlap':r['cutoff_kept_mass']*r['folded_fidelity'],
                  'optimized_full_overlap_if_sampled':(r['cutoff_kept_mass']*r['optimized_fidelity']
                    if r['optimized_fidelity'] is not None else None),
                  'support_full_overlap_ceiling':r['cutoff_kept_mass']*r['support_upper_bound'],
                  'folded_meets_fidelity':r['folded_fidelity']>=args.fidelity_min,
                  'optimized_meets_fidelity_if_sampled':(r['optimized_fidelity']>=args.fidelity_min
                    if r['optimized_fidelity'] is not None else None),
                  'passes_mass':r['cutoff_lost_mass']<=args.max_lost_mass,
                  'passes_full_overlap':r['cutoff_kept_mass']*r['folded_fidelity']>=args.full_fidelity_min})
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'boundary_samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(enriched[0]));w.writeheader();w.writerows(enriched)
    with (args.output/'cost_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(costs[0]));w.writeheader();w.writerows(costs)
    (args.output/'boundary_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    plot(rows,costs,args.output,args.fidelity_min,args.max_lost_mass,args.full_fidelity_min)
    for branch,b in results['branches'].items():
        print('\n'+branch)
        for key in ('folded_fidelity_witness','optimized_sample_witness','mass_criterion',
                    'folded_full_overlap','support_ceiling_necessary','cost_break_even'):
            print(' ',key,':',b[key]['first_failure_bracket'])
    print('\nCost comparator:',baseline_origin,'; reference:',base_ref,'; transport:',meta['fixed_single_pass_cx'])
    print('Results:',args.output)


if __name__=='__main__':
    main()