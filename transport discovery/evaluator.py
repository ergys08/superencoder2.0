"""Independent numerical assessment of a proposed transport architecture.

Never allow an architecture-generating LLM to edit this file. We report:
(1) linearised coverage at the fixed reference state;
(2) per-target oracle fidelity under a fixed optimiser budget;
(3) held-out NN fidelity evaluated by learner.py.

'Oracle' means the BEST FOUND in limited optimisation, NOT a proven global
optimum. Poor oracle fidelity does not prove a representational impossibility.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from primitives import apply_numpy, real_action, simulate, reference_tangents
from config import BOUNDS, Settings
from candidate import Candidate
from data import amplitude


def fidelity(target, prepared) -> float:
    """Pure-state overlap squared; works for real or complex amplitudes."""
    return float(abs(np.vdot(target,prepared))**2)


def reference_parameter_tangents(s: Settings, ref_eta: np.ndarray,
                                 reference: np.ndarray, fraction: float = 0.002,
                                 bin_edges: np.ndarray | None = None) -> np.ndarray:
    """Finite-difference derivatives w.r.t. *normalised* six Heston inputs.

    A unit of normalised parameter corresponds to the original range width.
    The step is 0.002 of that width. Valid in the interior of the range.
    """
    cols=[]
    for j in range(6):
        h=fraction*(BOUNDS[j,1]-BOUNDS[j,0])
        lo=ref_eta.copy();hi=ref_eta.copy()
        lo[j]-=h;hi[j]+=h
        vec=(amplitude(hi,s.n_qubits,s.n_fft,bin_edges)-amplitude(lo,s.n_qubits,s.n_fft,bin_edges))/(2*fraction)
        # Remove tiny norm/phase differentiation noise.
        vec=vec - reference*np.dot(reference,vec)
        cols.append(vec)
    return np.stack(cols,axis=1)


def tangent_residual(architecture: Candidate, reference: np.ndarray,
                     targets: np.ndarray) -> float:
    """Frobenius residual of six normalised Heston tangent vectors.

    Test whether the candidate's first-order directions at zero angles span
    derivatives of the Heston state with respect to its six parameters.
    Only local information at the reference, NOT global expressivity.
    """
    span=reference_tangents(reference,architecture.primitives)
    projection=np.linalg.lstsq(span,targets,rcond=1e-10)[0]
    return float(np.linalg.norm(targets-span@projection) / np.linalg.norm(targets))


def objective_and_jac(theta: np.ndarray, primitives, target, ref):
    """Exact fidelity gradient by reverse propagation through Pauli rotations.

    Forward: psi_{k+1}=U_k psi_k.
    Backward: b_k=U_{k+1}^T ... U_{L-1}^T target.
    d overlap/d theta_k = 1/2 b_k^T (-i P_k) psi_{k+1}.
    For odd-Y Pauli gates everything here is real.
    """
    states=[ref]
    for p,x in zip(primitives,theta):
        states.append(apply_numpy(states[-1],p,float(x)))
    overlap=float(np.dot(target,states[-1]))
    back=target
    gradient=np.empty(len(primitives),dtype=float)
    for k in reversed(range(len(primitives))):
        ind,sign=real_action(primitives[k].word)
        grad_overlap=.5*float(np.dot(back,states[k+1][ind]*sign))
        gradient[k]=-2*overlap*grad_overlap
        back=apply_numpy(back,primitives[k],-float(theta[k]))
    return 1-overlap*overlap, gradient


@dataclass
class OracleScores:
    mean_fidelity: float
    p05_fidelity: float
    minimum_fidelity: float
    mean_function_calls: float
    per_state_fidelity: list[float]


def oracle_evaluate(c: Candidate, ref: np.ndarray, targets: np.ndarray,
                    *, maxiter: int, restarts: int, seed: int) -> OracleScores:
    """Diagnose expressivity with a limited FAIR optimiser budget.

    All candidate architectures face the same targets and restart seeds.
    Each target is independently optimised; these parameters are *not* used
    at deployment and are *not* used to score NN generalisation.
    """
    rng=np.random.default_rng(seed)
    prims=c.primitives
    results=[];evals=[]
    for target in targets:
        best=0.
        calls=0
        for r in range(restarts):
            theta0=np.zeros(len(prims)) if r==0 else rng.normal(0.,.3,len(prims))
            result=minimize(objective_and_jac,theta0,args=(prims,target,ref),
                            method='L-BFGS-B',jac=True,
                            options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-7})
            current=fidelity(target,simulate(ref,prims,result.x))
            best=max(best,current)
            calls+=int(result.nfev)
        results.append(best);evals.append(calls)
    a=np.asarray(results)
    return OracleScores(float(a.mean()),float(np.quantile(a,.05)),float(a.min()),
                        float(np.mean(evals)),results)


def summarise_fidelities(ref: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Full distribution of fidelities with and without applying transport."""
    f=np.asarray([fidelity(a,b) for a,b in zip(targets,predictions)])
    identity=np.square(np.abs(targets@ref))
    return {
        'mean_fidelity':float(f.mean()),
        'p05_fidelity':float(np.quantile(f,.05)),
        'worst_fidelity':float(f.min()),
        'fraction_above_0999':float(np.mean(f>=.999)),
        'fraction_above_09999':float(np.mean(f>=.9999)),
        'identity_mean_fidelity':float(identity.mean()),
        'identity_p05_fidelity':float(np.quantile(identity,.05)),
        'mean_infidelity_reduction':float((1-f.mean())/(1-identity.mean())) if identity.mean()<1 else float('nan'),
    }