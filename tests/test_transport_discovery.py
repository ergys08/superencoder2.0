"""Transport discovery unit and integration tests (no model API required)."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'transport discovery'
for path in (str(ROOT),str(DIR)):
    if path not in sys.path:sys.path.insert(0,path)
from primitives import Primitive,apply_numpy,simulate,real_action,compile_qiskit,primitive_pool
from candidate import Candidate,mutate,seed_candidates
from evaluator import objective_and_jac,tangent_residual
from learner import TransportNN
from config import Settings,REFERENCE,BOUNDS
from openevolve_evaluator import parse_words


def pauli(word):
    mats={'I':np.eye(2,dtype=complex),
          'X':np.array([[0,1],[1,0]],complex),
          'Y':np.array([[0,-1j],[1j,0]],complex),
          'Z':np.diag([1,-1]).astype(complex)}
    out=np.array([[1]],complex)
    for p in reversed(word):out=np.kron(out,mats[p])
    return out


@pytest.mark.parametrize('word',['YII','IYZ','YZY','YYY','XYX','ZYZ'])
def test_real_rotation_matches_exact_matrix(word):
    if word.count('Y')%2!=1:pytest.skip('Odd-Y primitives only')
    rng=np.random.default_rng(15)
    state=rng.normal(size=2**len(word));state/=np.linalg.norm(state)
    theta=.31
    actual=apply_numpy(state,Primitive(word),theta)
    expected=expm(-.5j*theta*pauli(word))@state
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(actual),1.,atol=1e-12)


def test_order_identity_and_gradient():
    rng=np.random.default_rng(76)
    ref=rng.random(8);ref/=np.linalg.norm(ref)
    target=rng.random(8);target/=np.linalg.norm(target)
    prims=[Primitive('YII'),Primitive('ZYI'),Primitive('IYZ')]
    np.testing.assert_allclose(simulate(ref,prims,np.zeros(3)),ref)
    angles=rng.normal(size=3)*.4
    obj,grad=objective_and_jac(angles,prims,target,ref)
    eps=1e-6
    finite=[]
    for j in range(3):
        d=np.zeros(3);d[j]=eps
        plus=objective_and_jac(angles+d,prims,target,ref)[0]
        minus=objective_and_jac(angles-d,prims,target,ref)[0]
        finite.append((plus-minus)/(2*eps))
    np.testing.assert_allclose(grad,finite,atol=1e-7)


def test_reference_tangent_span():
    ref=np.ones(8)/np.sqrt(8)
    c=Candidate(('YII','IYI','IIY'))
    from primitives import reference_tangents
    w=reference_tangents(ref,c.primitives)
    assert tangent_residual(c,ref,w[:,:2])<1e-12


def test_candidate_mutations_obey_constraints():
    p=primitive_pool(4,max_support=3,full_two_qubit=True)
    seeds=seed_candidates(4,3,12,12)
    assert seeds
    parent=seeds[0];rng=np.random.default_rng(17)
    for _ in range(30):
        parent=mutate(parent,p,rng,max_cx=12,max_primitives=12)
        parent.validate(4,12,12)


def test_neural_anchors_identity():
    import torch
    model=TransportNN(Candidate(('YII','IYI')),width=8)
    x=torch.as_tensor(REFERENCE[None,:],dtype=torch.float64)
    np.testing.assert_allclose(model.angles(x).detach().numpy(),0.,atol=1e-14)
    psi=torch.as_tensor(np.ones((1,8))/np.sqrt(8),dtype=torch.float64)
    np.testing.assert_allclose(model(x,psi).detach().numpy(),psi.numpy(),atol=1e-12)


def test_original_parameter_ranges():
    np.testing.assert_allclose(REFERENCE,BOUNDS.mean(axis=1))
    assert tuple(BOUNDS[2]) == (.8,4.)
    assert tuple(BOUNDS[5]) == (-.9,-.1)


def test_safe_llm_candidate_parse(tmp_path):
    good=tmp_path/'good.py';good.write_text("WORDS = ['YII', 'IYI']\n")
    assert parse_words(good).words==('YII','IYI')
    bad=tmp_path/'bad.py';bad.write_text("import os\nWORDS=['YII']\n")
    with pytest.raises(ValueError):parse_words(bad)


def test_qiskit_compiler_matches_simulator_if_available():
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    ref=np.array([.1,.2,.1,.1,.3,.2,.2,.1]);ref/=np.linalg.norm(ref)
    prims=[Primitive('ZYI'),Primitive('IYX'),Primitive('YII'),Primitive('YYY')]
    angles=np.array([.4,-.3,.7,.1])
    qc=compile_qiskit(prims,angles)
    actual=Statevector(ref).evolve(qc).data
    expected=simulate(ref,prims,angles)
    np.testing.assert_allclose(actual,expected,atol=1e-11)
