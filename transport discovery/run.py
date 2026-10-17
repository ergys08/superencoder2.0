"""End-to-end fixed-reference transport architecture-discovery experiment.

Run from the repository root:
    python "transport discovery/run.py" --quick
    python "transport discovery/run.py" --output "transport discovery/results/full"
    python "transport discovery/run.py" --dataset ../optimal-transport/results/dataset.npz

This script deliberately separates architecture capacity, per-target oracle
optimisation (diagnostic only), and NN angle prediction (deployable method).
The final test states are never used in architecture selection or NN fitting.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import time
import numpy as np
from config import Settings
from data import create_dataset, save_dataset_metadata
from primitives import primitive_pool, compiled_cx_count
from candidate import Candidate, seed_candidates, mutate, save_candidate
from evaluator import (reference_parameter_tangents,tangent_residual,
                       oracle_evaluate,summarise_fidelities)
from learner import train,predict,save_model


def write_csv(path:Path,rows:list[dict]):
    if not rows:return
    names=list(dict.fromkeys(key for r in rows for key in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names)
        writer.writeheader();writer.writerows(rows)


def build_candidates(s,reference,tangents,external=None):
    """Greedy diversity-preserving baseline search over ordered primitives.

    A proposal is ranked by reference-state tangent coverage, with a weak
    gate-cost penalty. This cheap score is a heuristic, not guaranteed to
    correlate with the best finite-distance transport fidelity.
    """
    rng=np.random.default_rng(s.seed+193)
    pool=primitive_pool(s.n_qubits,s.max_support,full_two_qubit=True)
    archive={}
    records=[]

    def submit(c:Candidate,source):
        if c.words in archive:return
        try:c.validate(s.n_qubits,s.max_cx,s.max_primitives)
        except ValueError:return
        residual=tangent_residual(c,reference,tangents)
        # Sorting ONLY; no held-out states are used here.
        cost=c.cx_upper_bound
        score=residual+.045*(cost/max(1,s.max_cx))
        archive[c.words]=(score,c)
        records.append({'id':len(records),'source':source,'words':' '.join(c.words),
                        'n_parameters':c.n_angles,'cx_upper_bound':cost,
                        'reference_tangent_residual':residual,'screen_score':score})

    for c in seed_candidates(s.n_qubits,s.max_support,s.max_cx,s.max_primitives):submit(c,'seed')
    if external:
        raw=json.loads(Path(external).read_text())
        raw=raw['architectures'] if isinstance(raw,dict) and 'architectures' in raw else raw
        if not isinstance(raw,list):raise ValueError('Proposal JSON must be a list')
        for item in raw:
            try:submit(Candidate(tuple(item['words'])),'external')
            except (KeyError,TypeError,ValueError):continue
    if not archive:raise RuntimeError('No legal seed candidates')

    while len(records)<s.n_candidates:
        rank=sorted(archive.values(),key=lambda t:t[0])[:max(3,min(15,len(archive)))]
        parent=rank[int(rng.integers(len(rank)))][1] if rng.random()<.80 else list(archive.values())[int(rng.integers(len(archive)))][1]
        child=mutate(parent,pool,rng,s.max_cx,s.max_primitives)
        submit(child,'mutation')
        if len(archive)>=30000:break
    return records,archive


def oracle_sort_key(r,s):
    # Resource-aware, but primarily oracle fidelity on identical training targets.
    return -(r['oracle_mean_fidelity']-.025*r['cx_upper_bound']/max(1,s.max_cx))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='transport discovery/results',help='Folder for results; ignored by .gitignore')
    p.add_argument('--dataset',default=None,help='Optional original optimal-transport/results/dataset.npz')
    p.add_argument('--proposals',default=None,help='Optional list of LLM-proposed candidate dictionaries')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--n-qubits',type=int,default=8)
    p.add_argument('--n-candidates',type=int,default=None)
    p.add_argument('--max-cx',type=int,default=60)
    p.add_argument('--quick',action='store_true',help='Fast smoke test; NOT research-grade results')
    args=p.parse_args()
    s=Settings(seed=args.seed,n_qubits=args.n_qubits,max_cx=args.max_cx)
    if args.quick:
        s=replace(s,n_fft=2048,n_train=40,n_validation=12,n_test=12,
                  n_candidates=12,n_keep_oracle=4,n_finalists=2,
                  oracle_states=4,oracle_maxiter=25,oracle_restarts=1,
                  nn_epochs=14,nn_patience=6,nn_width=24,nn_batch=16,
                  max_primitives=12)
    if args.n_candidates is not None:s=replace(s,n_candidates=args.n_candidates)
    s.validate()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    print('Building fixed-reference Heston dataset...',flush=True)
    d=create_dataset(s,args.dataset)
    save_dataset_metadata(s,out,args.dataset)
    (out/'settings.json').write_text(json.dumps(asdict(s),indent=2))
    np.savez_compressed(out/'reference.npz',eta=d.eta_ref,amplitudes=d.psi_ref)
    # The initial reference state can be prepared by a deterministic StatePreparation
    # circuit. Its cost is constant and NOT included in transport CX budgets.
    print('Computing true Heston tangent directions at reference...',flush=True)
    jac=reference_parameter_tangents(s,d.eta_ref,d.psi_ref)
    print('Searching fixed primitive sequences...',flush=True)
    screen,archive=build_candidates(s,d.psi_ref,jac,args.proposals)
    write_csv(out/'screening.csv',screen)
    by_sig={tuple(r['words'].split()):r for r in screen}
    sorted_screen=sorted(screen,key=lambda r:r['screen_score'])
    selected=sorted_screen[:min(s.n_keep_oracle,len(sorted_screen))]
    oracle_targets=d.psi_train[:min(len(d.psi_train),s.oracle_states)]
    oracle_rows=[]
    print('Running bounded per-target oracle diagnostic...',flush=True)
    for k,r in enumerate(selected):
        candidate=Candidate(tuple(r['words'].split()))
        score=oracle_evaluate(candidate,d.psi_ref,oracle_targets,maxiter=s.oracle_maxiter,
                              restarts=s.oracle_restarts,seed=s.seed)
        cx,depth=compiled_cx_count(candidate.primitives)
        oracle_rows.append({**r,'oracle_mean_fidelity':score.mean_fidelity,
                            'oracle_p05_fidelity':score.p05_fidelity,
                            'oracle_min_fidelity':score.minimum_fidelity,
                            'oracle_avg_objective_calls':score.mean_function_calls,
                            'compiled_cx':cx,'compiled_depth':depth})
        print(f'  {k+1}/{len(selected)}: oracle mean F={score.mean_fidelity:.6f}, '
              f'tangent residual={r["reference_tangent_residual"]:.4f}, CX={cx}',flush=True)
    write_csv(out/'oracle.csv',oracle_rows)
    final_arches=sorted(oracle_rows,key=lambda r:oracle_sort_key(r,s))[:s.n_finalists]
    val_rows=[]
    trained=[]
    print('Training amortised parameter->angle networks on FINALISTS only...',flush=True)
    for k,r in enumerate(final_arches):
        candidate=Candidate(tuple(r['words'].split()))
        fit=train(candidate,d.eta_train,d.psi_train,d.eta_val,d.psi_val,d.psi_ref,
                  seed=s.seed,epochs=s.nn_epochs,width=s.nn_width,
                  batch_size=s.nn_batch,patience=s.nn_patience)
        val_pred,_=predict(fit.model,d.eta_val,d.psi_ref)
        val=summarise_fidelities(d.psi_ref,d.psi_val,val_pred)
        row={**r,**{'val_'+key:v for key,v in val.items()},
             'nn_epochs':fit.epochs_run}
        val_rows.append(row)
        trained.append((candidate,fit,row))
        print(f'  {k+1}/{len(final_arches)}: val mean F={val["mean_fidelity"]:.6f}, '
              f'val p05={val["p05_fidelity"]:.6f}, CX={r["compiled_cx"]}',flush=True)
    write_csv(out/'validation.csv',val_rows)
    # Fidelity feasibility is checked on validation only. Prefer cheapest feasible;
    # if nothing passes, report FAILURE and choose maximal p05 performance.
    feasible=[item for item in trained if item[2]['val_p05_fidelity']>=s.target_fidelity]
    if feasible:
        winner=min(feasible,key=lambda z:(z[2]['compiled_cx'], -z[2]['val_mean_fidelity']))
    else:
        winner=max(trained,key=lambda z:(z[2]['val_p05_fidelity'],z[2]['val_mean_fidelity']))
    candidate,fit,best=winner
    save_candidate(candidate,out/'winner.json')
    save_model(fit,candidate,out/'winner_model')
    # The test split is accessed ONCE, after all choices are frozen.
    test_pred,test_angles=predict(fit.model,d.eta_test,d.psi_ref)
    test=summarise_fidelities(d.psi_ref,d.psi_test,test_pred)
    np.savez_compressed(out/'heldout_predictions.npz',
                        eta=d.eta_test,target=d.psi_test,predicted=test_pred,
                        angles=test_angles)
    # Optionally measure an exact baseline with Qiskit, if installed.
    base={'reference_cx':None,'reference_depth':None,
          'direct_target_cx':None,'direct_target_depth':None}
    try:
        from qiskit import QuantumCircuit,transpile
        from qiskit.circuit.library import StatePreparation
        for tag,psi in [('reference',d.psi_ref),('direct_target',d.psi_val[0])]:
            qc=QuantumCircuit(s.n_qubits)
            qc.append(StatePreparation(psi),range(s.n_qubits))
            compiled=transpile(qc,basis_gates=['u','cx'],optimization_level=2)
            base[f'{tag}_cx']=int(compiled.count_ops().get('cx',0))
            base[f'{tag}_depth']=int(compiled.depth())
    except ImportError:
        pass
    report={'status':'fidelity_threshold_passed' if feasible else 'no_candidate_met_validation_threshold',
            'settings':asdict(s),'reference_parameters':d.eta_ref.tolist(),
            'states':{'train':len(d.eta_train),'validation':len(d.eta_val),'test':len(d.eta_test)},
            'selected_architecture':candidate.to_dict(),
            'selected_validation_metrics':best,
            'test_metrics':test,
            'reference_and_direct_baseline_costs':base,
            'elapsed_seconds':round(time.perf_counter()-t0,2),
            'limitations':[
                'Search space restricts to odd-Y Pauli rotations of bounded support.',
                'Search is local evolutionary mutation unless optional OpenEvolve/LLM is configured.',
                'Reference tangents test local first-order coverage only.',
                'Oracle optimiser is finite-budget and does not prove globally optimal fidelity.',
                'A learned NN is necessary for parameter-only transport without target-specific optimisation.',
                'Only the predefined fixed-maturity and fixed-S0 domain is tested.',
                'CNOT cost excludes one-time reference preparation; report both to compare total cost.',
                'Finite FFT/binning errors and precision can affect nominal fidelity.',
                'No hardware noise, connectivity routing or dynamical error correction is simulated.',
            ]}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('\nFINAL HELD-OUT TEST:',json.dumps({'status':report['status'],
          'test':test,'transport_cx':best['compiled_cx'],
          'reference_cx':base['reference_cx'],'winner':candidate.to_dict()},indent=2),flush=True)
    print('Saved:',out)


if __name__=='__main__':
    main()
