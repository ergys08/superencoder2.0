"""Oracle-fit a saved reference-anchored transport basis to ANY Heston model.

Examples (run from repository root):
  python -m five_mode_transport.fit_transport \
    --basis five_mode_transport/results/main/transport_basis.npz \
    --eta 0.05 0.04 2.0 0.04 0.30 -0.70
  python -m five_mode_transport.fit_transport \
    --basis five_mode_transport/results/main/transport_basis.npz \
    --eta 0.05 0.04 2.0 0.04 0.30 -0.70 --modes 5

This computes TRUE target quantiles from the Heston model and projects their
reference-relative displacement onto the stored learned modes. There is NO
neural network, no per-target gate optimisation, and no quantum circuit.
Out-of-box models can be fitted but are labelled EXTRAPOLATION; their fidelity
must be measured, not inferred from training accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .heston_quantiles import (
    QuantileConfig, PARAMETER_NAMES, heston_quantiles,
    quantiles_to_probabilities, validate_eta,
)
from .discover_transport_modes import SCHEMA_VERSION, fidelity, project


def load_basis(path: str|Path)->dict:
    """Load only numeric arrays (no pickle / executable code)."""
    with np.load(path,allow_pickle=False) as z:
        required={'schema_version','reference_eta','reference_quantiles',
                  'reference_probabilities','bounds','u_grid','bin_edges',
                  'weights','modes','selected_modes','s0','maturity',
                  'n_fft','n_quantiles','n_bins','x_half_width',
                  'min_retained_mass','max_negative_mass'}
        missing=required-set(z.files)
        if missing:
            raise ValueError(f'Incompatible basis: missing {sorted(missing)}')
        basis={k:np.array(z[k]).copy() for k in z.files}
    if int(basis['schema_version'])!=SCHEMA_VERSION:
        raise ValueError('Unknown basis schema version')
    if basis['modes'].ndim!=2 or basis['modes'].shape[1]!=len(basis['u_grid']):
        raise ValueError('Basis modes and u_grid have inconsistent dimensions')
    return basis


def out_of_range(eta:np.ndarray,bounds:np.ndarray)->dict:
    """Flag each coordinate outside the requested discovery box.

    Report excess in units of its coordinate's full original box width.
    This is different from a radial distance from the reference.
    """
    width=bounds[:,1]-bounds[:,0]
    norm_under=np.maximum((bounds[:,0]-eta)/width,0.)
    norm_over=np.maximum((eta-bounds[:,1])/width,0.)
    excursions=norm_under+norm_over
    return {'within_fitted_parameter_bounds':bool(np.all(excursions<=1e-12)),
            'outside_parameters':[PARAMETER_NAMES[i] for i in np.flatnonzero(excursions>1e-12)],
            'normalised_boundary_excess':dict(zip(PARAMETER_NAMES,map(float,excursions)))}


def fit_model(eta:np.ndarray|list[float],basis:dict,n_modes:int|None=None)->dict:
    """Return reference-to-target oracle projection, approximate state and diagnostics.

    The source/reference quantile Q_ref is FIXED. Any Q_eta is computed from
    the actual Heston characteristic function using the stored FFT settings.
    The coefficients are weighted projections of Q_eta-Q_ref, so the zero-mode
    model is simply the reference distribution (up to rebinning artifacts).
    """
    eta=validate_eta(eta)
    k=int(basis['selected_modes']) if n_modes is None else int(n_modes)
    modes=basis['modes']
    if k<0 or k>len(modes):
        raise ValueError(f'n_modes must be in [0,{len(modes)}]')
    cfg=QuantileConfig(s0=float(basis['s0']),maturity=float(basis['maturity']),
                       n_fft=int(basis['n_fft']),n_quantiles=int(basis['n_quantiles']),
                       n_bins=int(basis['n_bins']),x_half_width=float(basis['x_half_width']),
                       min_retained_mass=float(basis['min_retained_mass']),
                       max_negative_mass=float(basis['max_negative_mass']))
    actual=heston_quantiles(eta,cfg)
    if not np.allclose(actual.bin_edges,basis['bin_edges'],atol=1e-12,rtol=0):
        raise ValueError('Inconsistent log-price bin grid; cannot compare amplitudes')
    if not np.allclose(actual.u,basis['u_grid'],atol=1e-14,rtol=0):
        raise ValueError('Inconsistent quantile grid')
    reference_q=basis['reference_quantiles']
    q_hat,coeff=project(reference_q,actual.quantiles,modes,basis['weights'],k)
    p_hat,q_mono,was_corrected=quantiles_to_probabilities(q_hat,actual.u,actual.bin_edges)
    F=fidelity(actual.probabilities,p_hat)
    w2=float(np.sqrt(np.sum(basis['weights']*(actual.quantiles-q_mono)**2)))
    out=out_of_range(eta,basis['bounds'])
    reference_eta=basis['reference_eta']
    norm_r=float(np.max(np.abs(eta-reference_eta)/(basis['bounds'][:,1]-basis['bounds'][:,0])))
    r1=reference_q
    baseline_probs,_,_=quantiles_to_probabilities(r1,actual.u,actual.bin_edges)
    base_fidelity=fidelity(actual.probabilities,baseline_probs)
    if not out['within_fitted_parameter_bounds']:
        warning=('EXTRAPOLATION: Heston parameters fall outside the region used to '
                 'fit and validate transport modes. Accuracy cannot be assumed.')
    else:
        warning=None
    result={'parameters':dict(zip(PARAMETER_NAMES,map(float,eta))),
            'reference_parameters':dict(zip(PARAMETER_NAMES,map(float,reference_eta))),
            'number_of_modes':k,
            'normalised_linf_distance_from_reference':norm_r,
            'oracle_coefficients':coeff.tolist(),
            'amplitude_fidelity':F,'infidelity':1-F,
            'weighted_quantile_w2_error':w2,
            'identity_transport_fidelity':base_fidelity,
            'isotonic_correction_required':was_corrected,
            **out,'warning':warning,
            'interpretation':'Oracle classical transport: target Heston quantiles were required to fit coefficients. Not a quantum circuit.'}
    return {'report':result,'true_quantiles':actual.quantiles,
            'fitted_quantiles':q_mono,
            'transport_source_log_prices':reference_q,
            'transport_target_log_prices':q_mono,
            'transport_log_price_displacements':q_mono-reference_q,
            'true_probabilities':actual.probabilities,
            'fitted_probabilities':p_hat,
            'fitted_amplitudes':np.sqrt(p_hat),
            'true_amplitudes':actual.amplitudes,
            'u_grid':actual.u,'bin_edges':actual.bin_edges}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basis',type=Path,required=True,help='Saved transport_basis.npz from discover_transport_modes')
    p.add_argument('--eta',nargs=6,type=float,required=True,
                   metavar=tuple(k.upper() for k in PARAMETER_NAMES))
    p.add_argument('--modes',type=int,default=None,help='Override the recommended number of modes')
    p.add_argument('--out',type=Path,default=Path('five_mode_transport/results/new_model'))
    args=p.parse_args()
    result=fit_model(args.eta,load_basis(args.basis),args.modes)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'fit_report.json').write_text(json.dumps(result['report'],indent=2))
    np.savez_compressed(args.out/'fit_states.npz',**{k:v for k,v in result.items() if k!='report'})
    print(json.dumps({'report_file':str(args.out/'fit_report.json'),
                      **result['report']},indent=2))


if __name__=='__main__':
    main()