#!/usr/bin/env python3
"""Finite multi-parameter Heston state transport along a path in log/atanh coordinates.

Path z(s)=(1-s) z_reference + s z_target, s in [0,1]. Fits a *shared*
Pauli-generator Hamiltonian to the exact target tangent at each path midpoint,
then propagates the transported state without resetting to exact intermediate
states. Reports path fidelity, local tangent leakage, and the integrated
coefficients needed for the cheaper single-pass approximation.

Run: python heston_coninuity/five_parameter_finite_transport.py --data FILE
"""
from __future__ import annotations
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from heston_calibrations import read_panel
from five_parameter_tangent_generators import (Settings,vector_to_z,z_to_parameters,jacobian,state,
    make_encoder,library,select_shared,diagnostics)
from v0_naive_tangent_generators import actions
from v0_finite_tangent_transport import matrix_of_generator


def fit_directions(psi,J,words):
    A=actions(psi,words)
    coeff=np.linalg.lstsq(A,J,rcond=1e-10)[0]
    relative=float(np.sum((J-A@coeff)**2)/max(np.sum(J*J),1e-15))
    return coeff,relative


def finite_path(encoder,z0,z1,words,n_steps=24,fd_step=0.0005):
    psi0,kept0,*_=state(encoder,z0)
    goal,kept1,*_=state(encoder,z1)
    transmitted=psi0.copy()
    matrices=[matrix_of_generator(w) for w in words]
    ds=1.0/n_steps
    dz=z1-z0
    angles=np.zeros(len(words))
    steps=[]
    for step_index in range(n_steps):
        s=(step_index+.5)*ds
        psi,J,retained,_=jacobian(encoder,z0+s*dz,fd_step)
        tangent=J@dz
        A=actions(psi,words)
        c=np.linalg.lstsq(A,tangent,rcond=1e-10)[0]
        residue=tangent-A@c
        relative=float(np.dot(residue,residue)/max(np.dot(tangent,tangent),1e-15))
        H=sum((float(coef)*K for coef,K in zip(c,matrices) if abs(coef)>1e-15),start=matrices[0]*0)
        transmitted=expm_multiply(ds*H,transmitted)
        angles+=ds*c
        ideal,_keep,*_=state(encoder,z0+(step_index+1)*ds*dz)
        f=float(np.dot(ideal,transmitted)**2)
        steps.append({'path_fraction':(step_index+1)*ds,'conditional_fidelity':f,
                      'local_relative_tangent_leakage':relative,'retained_mass':_keep})
    return {'final_state':transmitted,'goal':goal,'reference_state':psi0,
            'integrated_angles':angles,'step_diagnostics':steps,
            'final_fidelity':float(np.dot(goal,transmitted)**2),
            'reference_fidelity':float(np.dot(goal,psi0)**2),
            'kept_mass':kept1,
            'path_max_local_leakage':max(t['local_relative_tangent_leakage'] for t in steps)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path)
    p.add_argument('--reference-index',type=int,default=0)
    p.add_argument('--target-index',type=int,default=15)
    p.add_argument('--support',type=int,default=3)
    p.add_argument('--budget',type=int,default=16)
    p.add_argument('--steps',type=int,default=24)
    p.add_argument('--output',default='heston_coninuity/results/five_parameter_finite_transport')
    a=p.parse_args()
    panel=read_panel('2013',a.data,ROOT/'heston_calibration_results'/'data')
    enc=make_encoder(Settings(support=a.support,budget=a.budget))
    start=vector_to_z(panel.iloc[a.reference_index]);goal=vector_to_z(panel.iloc[a.target_index])
    psi,J,*_=jacobian(enc,start)
    words=select_shared(psi,J,library(8,a.support),a.budget)
    result=finite_path(enc,start,goal,words,a.steps)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    fields=list(result['step_diagnostics'][0]);
    with (out/'path_steps.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(result['step_diagnostics'])
    summary={'reference_date':str(panel.iloc[a.reference_index].date.date()),
      'target_date':str(panel.iloc[a.target_index].date.date()),
      'reference_params':z_to_parameters(start),'target_params':z_to_parameters(goal),
      'reference_identity_fidelity':result['reference_fidelity'],
      'finite_path_fidelity':result['final_fidelity'],
      'retained_target_mass':result['kept_mass'],
      'max_local_tangent_leakage':result['path_max_local_leakage'],
      'selected_words':words,'integrated_angles':result['integrated_angles'].tolist(),
      'number_of_exact_exponential_steps':a.steps,
      'note':'This trajectory uses oracle state derivatives at every midpoint; no gate compilation or NN.'}
    (out/'finite_path_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4.1)); ax.plot([0]+[r['path_fraction'] for r in result['step_diagnostics']],
        [1]+[r['conditional_fidelity'] for r in result['step_diagnostics']]);
    ax.set(xlabel='Fraction of simultaneous five-parameter path',ylabel='Conditional fidelity',title='Finite path: exact exponential of fitted local generators');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'finite_path_fidelity.png',dpi=165);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()