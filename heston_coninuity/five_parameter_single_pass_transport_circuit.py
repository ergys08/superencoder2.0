#!/usr/bin/env python3
"""Compile and test one-pass five-parameter Heston transport circuits.

Select a shared Pauli library at a reference, integrate midpoint tangent
coefficients along a simultaneous five-parameter path, then apply each Pauli
rotation ONCE. Optionally optimise these same fixed angles at the endpoint.
Gate metrics and OpenQASM exclude preparation of the reference state. All
endpoint fitting is an oracle optimisation, NOT a learned parameter-angle map.

Run: python heston_coninuity/five_parameter_single_pass_transport_circuit.py --data FILE
"""
from __future__ import annotations
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from heston_calibrations import read_panel
from five_parameter_tangent_generators import Settings,vector_to_z,jacobian,make_encoder,library,select_shared
from five_parameter_finite_transport import finite_path
from single_pass_transport_circuit import PassCircuit,compile_pass,fidelity
from finite_path_circuit_compilation import qasm_for_gates,simulate_gates


def single_pass(encoder,z0,z1,words,steps=24,maxiter=200):
    fp=finite_path(encoder,z0,z1,words,steps)
    circuit=PassCircuit(fp['reference_state'],words)
    folded=fp['integrated_angles']
    initial_state=circuit.propagate(folded)
    fitted,optimal_fid,iterations,success=circuit.endpoint_fit(fp['goal'],folded,None,maxiter)
    return fp,{'folded_angles':folded,'optimised_angles':fitted,
      'folded_fidelity':fidelity(fp['goal'],initial_state),
      'endpoint_fidelity':optimal_fid,'optimizer_iterations':iterations,
      'optimizer_success':success}


def compiler_cost(ref,words,theta,layout='linear'):
    circuit=compile_pass(words,theta,layout)
    gates=list(circuit.active())
    check=simulate_gates(ref,gates)
    return gates,circuit.metrics(),check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path)
    p.add_argument('--reference-index',type=int,default=0)
    p.add_argument('--target-index',type=int,default=15)
    p.add_argument('--support',type=int,default=3)
    p.add_argument('--budget',type=int,default=16)
    p.add_argument('--steps',type=int,default=24)
    p.add_argument('--maxiter',type=int,default=250)
    p.add_argument('--output',default='heston_coninuity/results/five_parameter_single_pass_transport_circuit')
    a=p.parse_args()
    panel=read_panel('2013',a.data,ROOT/'heston_calibration_results'/'data')
    enc=make_encoder(Settings(support=a.support,budget=a.budget))
    z0=vector_to_z(panel.iloc[a.reference_index]);z1=vector_to_z(panel.iloc[a.target_index])
    psi,J,*_=jacobian(enc,z0)
    words=select_shared(psi,J,library(8,a.support),a.budget)
    fp,ans=single_pass(enc,z0,z1,words,a.steps,a.maxiter)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    report={'reference_date':str(panel.iloc[a.reference_index].date.date()),
      'target_date':str(panel.iloc[a.target_index].date.date()),'selected_words':words,
      'identity_fidelity':fp['reference_fidelity'],
      'finite_path_fidelity':fp['final_fidelity'],
      'one_pass_folded_fidelity':ans['folded_fidelity'],
      'one_pass_endpoint_optimised_fidelity':ans['endpoint_fidelity'],
      'target_retained_mass':fp['kept_mass'],
      'folded_angles':ans['folded_angles'].tolist(),
      'optimised_angles':ans['optimised_angles'].tolist(),
      'local_optimisation_not_global_certification':True,
      'compiled':{}}
    for kind,angles in [('folded',ans['folded_angles']),('optimised',ans['optimised_angles'])]:
      for layout in ['linear','all_to_all']:
        gates,metrics,got=compiler_cost(fp['reference_state'],words,angles,layout)
        expected=PassCircuit(fp['reference_state'],words).propagate(angles)
        err=float(np.linalg.norm(got-expected))
        if err>1e-8:raise AssertionError(f'Compiler incorrect: {kind}/{layout}, err={err}')
        report['compiled'][kind+'_'+layout]={**metrics,'simulation_error_norm':err}
        (out/f'{kind}_{layout}.qasm').write_text(qasm_for_gates(gates,8))
    (out/'single_pass_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()