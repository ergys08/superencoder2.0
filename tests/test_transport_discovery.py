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

# ---- Robust objective / domain tests added in October 2026 ----

def test_all_64_corners_cover_exact_original_box():
    from robustness import corners
    c = corners()
    assert c.shape == (64, 6)
    assert len(np.unique(c,axis=0)) == 64
    assert ((c == BOUNDS[:,0]) | (c == BOUNDS[:,1])).all()
    np.testing.assert_allclose(c[0],BOUNDS[:,0])
    np.testing.assert_allclose(c[-1],BOUNDS[:,1])


def test_independent_random_domain_samples_are_seeded():
    from robustness import interior
    p = interior(20,seed=17)
    assert p.shape == (20,6)
    assert ((p>BOUNDS[:,0]) & (p<BOUNDS[:,1])).all()
    np.testing.assert_allclose(p, interior(20,17))
    assert not np.allclose(p,interior(20,18))


def test_robust_loss_hard_samples_receive_greater_weight():
    import torch
    from learner import robust_infidelity_loss
    t = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[1.,0.]],dtype=torch.float64)
    p = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]],dtype=torch.float64)
    ordinary = float(robust_infidelity_loss(p,t,tail_fraction=.25,tail_weight=0))
    risk = float(robust_infidelity_loss(p,t,tail_fraction=.25,tail_weight=.75))
    assert ordinary == pytest.approx(.25)
    assert risk == pytest.approx(.8125)


def test_robustness_audit_detects_corner_failure_and_reports_not_certified():
    from robustness import audit
    # Tiny 2-qubit Heston dataset ensures this test is cheap and independent.
    from data import amplitude
    s = Settings(n_qubits=2,n_fft=512,n_train=2,n_validation=2,n_test=2)
    ref = amplitude(REFERENCE,s.n_qubits,s.n_fft)
    class IdentityModel:
        def eval(self):return self
        def __call__(self,eta,reference):return reference.expand(len(eta),-1)
        def angles(self,eta):
            import torch
            return torch.zeros((len(eta),1),dtype=torch.float64)
    rep=audit(IdentityModel(),ref,s,seed=5,n_interior=3,
              adversarial_maxiter=0,adversarial_popsize=0)
    assert rep.n_corners == 64
    assert rep.n_interior == 3
    assert rep.adversarial_min_fidelity is None
    assert rep.observed_min_fidelity <= rep.observed_mean_fidelity <= 1
    assert rep.summary()['note'].endswith('NOT a global certificate.')


def test_rebinned_amplitudes_on_same_edges_as_reference():
    from data import amplitude
    from heston import HestonParameters, HestonModel
    p=HestonModel(HestonParameters(s0=100.,maturity=1.,
                                  mu=float(REFERENCE[0]),v0=float(REFERENCE[1]),
                                  kappa=float(REFERENCE[2]),theta=float(REFERENCE[3]),
                                  xi=float(REFERENCE[4]),rho=float(REFERENCE[5])))
    dist=p.terminal_distribution(n_qubits=3,n_fft=512)
    a=amplitude(REFERENCE,3,512,dist.log_price_edges)
    np.testing.assert_allclose(a,dist.amplitudes,rtol=0,atol=1e-10)


def test_proposal_extractor_never_executes_code(tmp_path):
    from extract_proposals import extract
    one=tmp_path/'a.py';one.write_text("WORDS=['YII','IYZ']")
    two=tmp_path/'b.py';two.write_text("WORDS=['YII','IYZ']")
    bad=tmp_path/'bad.py';bad.write_text('raise RuntimeError(\"must not execute\")')
    # Extractor expects 8-qubit words. The two 3-qubit examples should be
    # rejected safely; the unsafe file should not run either.
    result,report=extract([one,two,bad])
    assert result==[]
    assert report['rejected']==3


def test_proposal_extractor_accepts_valid_architecture(tmp_path):
    from extract_proposals import extract
    from openevolve_evaluator import parse_words
    source=ROOT/'transport discovery'/'llm_candidate.py'
    c=parse_words(source)
    proposals,stats=extract([source,source])
    assert stats['accepted']==1
    assert stats['rejected']==0
    assert proposals==[{'words':list(c.words)}]


def test_llm_provider_configs_have_no_embedded_secret():
    for name in ('openevolve_gemini.yaml','openevolve_openai.yaml'):
        contents=(DIR/name).read_text()
        assert 'api_key: ${' in contents
        assert 'enforce_evolve_blocks: true' in contents