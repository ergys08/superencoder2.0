"""OpenEvolve evaluator. Candidate program is PARSED, NEVER EXECUTED.

Usage (once OpenEvolve and a model provider are configured):
 openevolve-run 'transport discovery/llm_candidate.py' \
   'transport discovery/openevolve_evaluator.py' --iterations 40

The evaluator uses only the 48 reproducibly generated search-training states
and reference tangents plus 6 fixed box corners, never validation/test states. Its oracle is finite
budget. The end-to-end `run.py --proposals` must evaluate archived finalists
with an NN on validation, then one final held-out test.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
from pathlib import Path
import ast
import numpy as np
from config import Settings, BOUNDS
from data import create_dataset
from candidate import Candidate
from evaluator import reference_parameter_tangents,tangent_residual,oracle_evaluate
from primitives import primitive_pool


def parse_words(path):
    """Only accept a literal assignment WORDS=[...], no executable Python."""
    tree=ast.parse(Path(path).read_text())
    found=[]
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            continue  # docstring
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='WORDS':
            found.append(ast.literal_eval(node.value))
            continue
        raise ValueError('Candidate files can contain only WORDS = [literal strings]')
    if len(found)!=1 or not isinstance(found[0],list) or not all(isinstance(w,str) for w in found[0]):
        raise ValueError('Expected one WORDS list of strings')
    return Candidate(tuple(found[0]))


@lru_cache(maxsize=1)
def _fixed_training_data():
    s=Settings(n_train=48,n_validation=8,n_test=8,n_fft=2048,
               oracle_states=6,oracle_maxiter=45,oracle_restarts=1)
    data=create_dataset(s)
    tangents=reference_parameter_tangents(s,data.eta_ref,data.psi_ref)
    # Public corners are stress conditions, NOT held-out targets. Vary all six
    # parameters simultaneously. A small fixed subset limits API/search cost;
    # the full 64-corner and adversarial audit happens in run.py.
    bits = ((0,0,0,0,0,0),(1,1,1,1,1,1),
            (0,1,0,1,0,1),(1,0,1,0,1,0),
            (0,0,1,1,1,1),(1,1,0,0,0,0))
    from data import amplitude
    corner_states = np.stack([amplitude(np.array([BOUNDS[j, b] for j,b in enumerate(row)]),
                                        s.n_qubits,s.n_fft) for row in bits])
    return s,data,tangents,corner_states


def evaluate(program_path):
    """Return score and diagnostics in OpenEvolve-compatible dictionary."""
    try:
        candidate=parse_words(program_path)
        s,d,t,corners=_fixed_training_data()
        candidate.validate(s.n_qubits,s.max_cx,s.max_primitives)
        allowed={p.word for p in primitive_pool(s.n_qubits,s.max_support,full_two_qubit=True)}
        if not set(candidate.words).issubset(allowed):
            raise ValueError('Only approved primitive words are allowed')
        resid=tangent_residual(candidate,d.psi_ref,t)
        # Uniform training/difficult-corner oracle targets for EVERY proposal.
        # These target-specific angles only diagnose capacity, and are NOT the
        # deployable NN map. Run the full robust NN check on finalists later.
        targets=np.concatenate([d.psi_train[:s.oracle_states], corners],axis=0)
        oracle=oracle_evaluate(candidate,d.psi_ref,targets,
                               maxiter=s.oracle_maxiter,restarts=s.oracle_restarts,seed=s.seed)
        # Min-fidelity oriented heuristic. The gate penalty matters only after
        # reliable transformations have been found; keep raw Pareto metrics.
        score=max(0.,oracle.minimum_fidelity - .06*resid
                  - .02*candidate.cx_upper_bound/s.max_cx)
        return {'combined_score':score,
                'oracle_worst_fidelity':oracle.minimum_fidelity,
                'oracle_p05_fidelity':oracle.p05_fidelity,
                'oracle_mean_fidelity':oracle.mean_fidelity,
                'tangent_coverage':1-resid,
                'cx_upper_bound':float(candidate.cx_upper_bound),
                'number_of_gates':float(candidate.n_angles)}
    except Exception:
        return {'combined_score':0.,'oracle_worst_fidelity':0.,
                'oracle_p05_fidelity':0.,'oracle_mean_fidelity':0.,'tangent_coverage':0.,
                'cx_upper_bound':999.,'number_of_gates':999.}