#!/usr/bin/env python3
"""Five-parameter Heston tangent vectors and a shared low-support Pauli library.

Use: python heston_coninuity/five_parameter_tangent_generators.py

The parameter coordinates are (log v0, log kappa, log theta, log xi,
atanh rho). Central differences are performed in these coordinates; hence the
five tangent vectors automatically respect positivity and |rho|<1.
Selection uses ONLY the reference state's five tangent vectors, independently
normalised for a balanced initial architecture, and no future calibrations.
A low *local* residual is not evidence of good finite-path transport.
"""
from __future__ import annotations
import argparse, csv, json, sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from heston_calibrations import FixedHestonEncoder, read_panel, PARAMETERS
from v0_naive_tangent_generators import pauli_words, actions

COORDS=['log_v0','log_kappa','log_theta','log_xi','atanh_rho']

@dataclass
class Settings:
    qubits:int=8
    fft_size:int=16384
    fft_halfwidth:float=3.5
    cutoff:float=0.7
    maturity:float=1.0
    drift:float=0.05
    support:int=3
    budget:int=16
    fd_step:float=0.0005
    reference_index:int=0
    output:str='heston_coninuity/results/five_parameter_tangent_generators'


def vector_to_z(row):
    vals=np.array([float(row[k]) for k in PARAMETERS],dtype=float)
    if np.any(vals[:4]<=0) or abs(vals[4])>=1:
        raise ValueError('Five parameters must satisfy positive v0,kappa,theta,xi and |rho|<1')
    return np.r_[np.log(vals[:4]),np.arctanh(vals[4])]


def z_to_parameters(z):
    return dict(zip(PARAMETERS,[*np.exp(np.clip(z[:4],-20,20)),float(np.tanh(z[4]))]))


def state(encoder,z):
    return encoder.state(z_to_parameters(z))


def jacobian(encoder,z,fd_step=0.0005):
    psi,retained,negative,edge=state(encoder,z)
    jac=np.empty((len(psi),5))
    convergence=[]
    for k in range(5):
        d=np.eye(5)[k]*fd_step
        first=(state(encoder,z+d)[0]-state(encoder,z-d)[0])/(2*fd_step)
        half=(state(encoder,z+d/2)[0]-state(encoder,z-d/2)[0])/fd_step
        jac[:,k]=half
        convergence.append(float(np.linalg.norm(first-half)/max(np.linalg.norm(half),1e-12)))
    jac=jac-psi[:,None]*(psi@jac)[None,:]
    return psi,jac,retained,dict(fd_relative_changes=convergence,negative_mass=negative,fft_edge_mass=edge)


def library(qubits=8,support=3):
    groups=pauli_words(qubits,support)
    return [word for r in range(1,support+1) for word in groups[r]]


def residual(P,J,selected):
    if len(selected)==0:return float(np.sum(J*J)),np.zeros((0,J.shape[1]))
    c=np.linalg.lstsq(P[:,selected],J,rcond=1e-10)[0]
    r=J-P[:,selected]@c
    return float(np.sum(r*r)),c


def select_shared(psi,J,words,budget):
    """Greedy joint Frobenius tangent fit; avoids privileging one parameter."""
    norms=np.linalg.norm(J,axis=0)
    desired=J/np.maximum(norms,1e-12)
    A=actions(psi,words)
    A=A/np.maximum(np.linalg.norm(A,axis=0),1e-12)
    chosen=[]
    current=desired.copy()
    for i in range(min(budget,len(words))):
        corr=A.T@current
        scores=np.einsum('ij,ij->i',corr,corr)
        scores[chosen]=-np.inf
        ix=int(np.argmax(scores))
        if scores[ix]<1e-16:break
        chosen.append(ix)
        _,c=residual(A,desired,chosen)
        current=desired-A[:,chosen]@c
    return [words[i] for i in chosen]


def diagnostics(psi,J,selected):
    A=actions(psi,selected)
    r,c=residual(A,J,list(range(len(selected))))
    errs=J-A@c
    ns=np.sum(J*J,axis=0)
    return {'relative_total_leakage':float(r/max(np.sum(ns),1e-15)),
            'relative_each_tangent_leakage':dict(zip(COORDS,(np.sum(errs*errs,axis=0)/np.maximum(ns,1e-15)).tolist())),
            'tangent_norms':dict(zip(COORDS,np.sqrt(ns).tolist())),
            'state_tangent_orthogonality':float(np.max(np.abs(psi@J))),
            'rank_of_five_tangents':int(np.linalg.matrix_rank(J,tol=1e-9)),
            'generator_action_rank':int(np.linalg.matrix_rank(A,tol=1e-9))}


def make_encoder(s):
    return FixedHestonEncoder(qubits=s.qubits,n_fft=s.fft_size,
        fft_halfwidth=s.fft_halfwidth,cutoff=s.cutoff,maturity=s.maturity,drift=s.drift)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path)
    a.add_argument('--reference-index',type=int,default=0)
    a.add_argument('--support',type=int,default=3)
    a.add_argument('--budget',type=int,default=16)
    a.add_argument('--output',default=Settings.output)
    args=a.parse_args()
    s=Settings(reference_index=args.reference_index,support=args.support,budget=args.budget,output=args.output)
    panel=read_panel('2013',args.data,ROOT/'heston_calibration_results'/'data')
    ref=panel.iloc[s.reference_index]
    encoder=make_encoder(s)
    psi,J,kept,info=jacobian(encoder,vector_to_z(ref),s.fd_step)
    words=library(s.qubits,s.support)
    selected=select_shared(psi,J,words,s.budget)
    report={'reference_date':str(ref.date.date()),'reference_parameters':{k:float(ref[k]) for k in PARAMETERS},
            'retained_probability':kept,'all_candidates':len(words),'selected_generators':selected,
            'selection_uses_future_dates':False,'derivative_check':info,'diagnostics':diagnostics(psi,J,selected),
            'settings':asdict(s)}
    out=Path(s.output);out.mkdir(parents=True,exist_ok=True)
    (out/'generator_selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()