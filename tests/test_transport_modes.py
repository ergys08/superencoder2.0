"""Invariant tests + a compact Heston end-to-end regression test."""
import json
import numpy as np
import pytest
from transport_modes.heston_quantiles import (
    QuantileConfig,ORIGINAL_BOUNDS,grid_weights,heston_quantiles,
    quantiles_to_probabilities,validate_eta,
)
from transport_modes.discover_transport_modes import (
    sample_shell,fit_reference_modes,project,corners,run,
)
from transport_modes.fit_transport import load_basis,fit_model,out_of_range

REF=ORIGINAL_BOUNDS.mean(axis=1)


def test_heston_reference_quantiles_and_probabilities():
    law=heston_quantiles(REF,QuantileConfig(n_fft=1024,n_quantiles=128))
    assert law.u.shape==(128,)
    assert law.quantiles.shape==(128,)
    assert law.probabilities.shape==(256,)
    assert np.all(np.diff(law.quantiles)>=0)
    assert np.isclose(law.probabilities.sum(),1,atol=1e-12)
    assert np.isclose(np.linalg.norm(law.amplitudes),1,atol=1e-12)


def test_heston_model_input_is_accepted():
    from heston import HestonModel,HestonParameters
    p=HestonParameters(**dict(zip(['mu','v0','kappa','theta','xi','rho'],REF)))
    a=heston_quantiles(HestonModel(p),QuantileConfig(n_fft=1024,n_quantiles=128))
    b=heston_quantiles(REF,QuantileConfig(n_fft=1024,n_quantiles=128))
    assert np.allclose(a.quantiles,b.quantiles)


def test_input_validation():
    with pytest.raises(ValueError):validate_eta([1,2])
    with pytest.raises(ValueError):validate_eta([.03,.02,2.,.04,.4,1.4])


def test_weighted_orthogonality_and_exact_reference_anchor():
    rng=np.random.default_rng(3)
    u=np.linspace(.01,.99,40)
    ref=np.linspace(-1,1,40)
    mat=ref+rng.normal(scale=.3,size=(25,40))
    modes,energies,_=fit_reference_modes(mat,ref,u,max_modes=8)
    gram=(modes*grid_weights(u))@modes.T
    assert np.allclose(gram,np.eye(8),atol=1e-10)
    q,c=project(ref,ref,modes,grid_weights(u),8)
    assert np.allclose(q,ref,atol=1e-11)
    assert np.allclose(c,0.,atol=1e-11)
    assert 0<energies.sum()<=1.+1e-12


def test_rank_1_projection_is_best_weighted_fit():
    u=np.linspace(.01,.99,20)
    ref=np.zeros(20)
    a=np.linspace(-1,1,20)
    b=np.sin(np.arange(20))
    train=np.stack([a,2*a,b,2*b])
    modes,_,_=fit_reference_modes(train,ref,u,2)
    target=a+.2*b
    q,coeff=project(ref,target,modes,grid_weights(u),1)
    assert coeff.shape==(1,)
    # Optimum along fixed mode has zero weighted residual derivative.
    assert abs(np.dot((target-q)*grid_weights(u),modes[0]))<1e-10


def test_shells_are_inside_and_exact_radius():
    rng=np.random.default_rng(77)
    for r in (0,.01,.05,.25,.5):
        pts=sample_shell(ORIGINAL_BOUNDS,REF,r,50,rng)
        assert np.all(pts>=ORIGINAL_BOUNDS[:,0]-1e-10)
        assert np.all(pts<=ORIGINAL_BOUNDS[:,1]+1e-10)
        dist=np.max(np.abs(pts-REF)/(ORIGINAL_BOUNDS[:,1]-ORIGINAL_BOUNDS[:,0]),axis=1)
        assert np.allclose(dist,r)
    assert len(corners(ORIGINAL_BOUNDS))==64


def test_isotonic_correction_reports_itself():
    u=np.linspace(.05,.95,10)
    q=np.array([0,.1,.2,.3,.4,.5,.3,.7,.8,.9])
    p,qfixed,was_corrected=quantiles_to_probabilities(q,u,np.linspace(0,1,17))
    assert was_corrected
    assert np.all(np.diff(qfixed)>=0)
    assert np.isclose(p.sum(),1)


def test_outside_range_flag():
    eta=REF.copy()
    assert out_of_range(eta,ORIGINAL_BOUNDS)['within_fitted_parameter_bounds']
    eta[1]=.12
    outside=out_of_range(eta,ORIGINAL_BOUNDS)
    assert not outside['within_fitted_parameter_bounds']
    assert outside['outside_parameters']==['v0']
    assert outside['normalised_boundary_excess']['v0']>0


def test_end_to_end_basis_and_fit(tmp_path):
    out=tmp_path/'study'
    result=run(reference=REF,bounds=ORIGINAL_BOUNDS,cfg=QuantileConfig(n_fft=1024,n_quantiles=64),
       out=out,n_train=14,n_validation=5,n_test=5,max_modes=5,
       n_per_radius=3,radii=[0,.02,.5],threshold=.999999,
       criterion='worst',seed=73)
    assert result['n_corners']==64
    assert (out/'mode_significance.csv').exists()
    assert (out/'fidelity_vs_radius.png').exists()
    assert (out/'transport_basis.npz').exists()
    assert len(json.loads((out/'summary.json').read_text())['radii'])==3
    basis=load_basis(out/'transport_basis.npz')
    # No knowledge of target is stored in the basis; target generated fresh.
    same=fit_model(REF,basis,n_modes=5)
    assert np.linalg.norm(same['report']['oracle_coefficients'])<1e-10
    assert same['report']['within_fitted_parameter_bounds']
    assert same['report']['amplitude_fidelity']>.98
    outside=REF.copy();outside[0]=.145
    fit=fit_model(outside,basis,n_modes=5)
    assert fit['report']['warning'].startswith('EXTRAPOLATION')
    assert fit['report']['outside_parameters']==['mu']
    assert 0<=fit['report']['amplitude_fidelity']<=1.+1e-10
    assert np.isclose(np.sum(fit['fitted_probabilities']),1.)


def test_uses_saved_historical_quantiles_and_preserves_split(tmp_path):
    """Original-study NPZ option must reuse supplied quantiles and bins."""
    from transport_modes.discover_transport_modes import sample_box
    cfg=QuantileConfig(n_fft=1024,n_quantiles=64)
    eta=sample_box(ORIGINAL_BOUNDS,12,10)
    laws=[heston_quantiles(e,cfg) for e in eta]
    first=laws[0]
    dataset=tmp_path/'dataset.npz'
    np.savez_compressed(dataset,
        params=eta,quantiles=np.stack([law.quantiles for law in laws]),
        probability_bins=np.stack([law.probabilities for law in laws]),
        u_grid=first.u,bin_edges=first.bin_edges,
        train_indices=np.arange(8),validation_indices=np.arange(8,10),
        test_indices=np.arange(10,12))
    summary=run(reference=REF,bounds=ORIGINAL_BOUNDS,cfg=cfg,
        out=tmp_path/'historic',n_train=3,n_validation=3,n_test=3,
        max_modes=5,n_per_radius=2,radii=[0,.05],
        seed=99,historical_dataset=dataset)
    assert summary['n_train']==8
    assert summary['n_validation']==2
    assert summary['n_test']==2
    assert summary['historical_dataset']==str(dataset)
    assert (tmp_path/'historic'/'transport_examples.npz').exists()
    assert (tmp_path/'historic'/'mode_functions.png').exists()
    assert '5' in summary['empirical_reach_by_mode_count']


def test_empirical_reach_requires_consecutive_shell_successes():
    from transport_modes.discover_transport_modes import empirical_reach
    data=[{'radius':0.,'n_modes':5,'min_fidelity':.99999},
          {'radius':.1,'n_modes':5,'min_fidelity':.99991},
          {'radius':.2,'n_modes':5,'min_fidelity':.9997},
          {'radius':.3,'n_modes':5,'min_fidelity':.99999}]
    assert empirical_reach(data,5,.9999)==.1