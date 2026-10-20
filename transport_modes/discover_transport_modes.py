"""Fit reference-anchored Heston quantile transport modes and test their reach.

This is an ORACLE CLASSICAL transport experiment, not a parameter->angle learner:
we compute a *target's true quantile function* before projecting it on the
basis. Results answer whether k modes represent target Heston distributions,
not whether a shallow quantum circuit implements those modes.

From repository root:
  python -m transport_modes.discover_transport_modes --out transport_modes/results/main
  python -m transport_modes.discover_transport_modes --quick
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from .heston_quantiles import (
    QuantileConfig, QuantileLaw, ORIGINAL_BOUNDS, PARAMETER_NAMES,
    bin_edges_for, grid_weights, heston_quantiles,
    quantiles_to_probabilities, validate_eta,
)

SCHEMA_VERSION = 1


def validate_box(bounds: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Bounds always have shape (6,2) in original Heston parameter order."""
    b = np.asarray(bounds, dtype=float)
    if b.shape != (6,2) or not np.isfinite(b).all() or np.any(b[:,1] <= b[:,0]):
        raise ValueError("bounds must be six finite [minimum,maximum] pairs")
    if np.any(reference < b[:,0]) or np.any(reference > b[:,1]):
        raise ValueError("Reference parameters must lie inside the requested box")
    for edge in (b[:,0],b[:,1]):
        validate_eta(edge)
    return b


def sample_box(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Independent seeded Latin-hypercube samples inside a 6D parameter box."""
    if n < 2:
        raise ValueError("Need at least two samples per training/validation/test split")
    u = qmc.LatinHypercube(d=6, seed=seed).random(n=n)
    return qmc.scale(u, bounds[:,0], bounds[:,1])


def sample_shell(bounds: np.ndarray, reference: np.ndarray, radius: float,
                 n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample exactly at normalized Linf radius r while staying INSIDE box.

    r=max_i abs(eta_i-reference_i)/(upper_i-lower_i). We pick a coordinate
    that can reach +/-r, fix it on the shell, and sample the remaining
    coordinates uniformly within their valid truncated ranges. Thus points
    are *not* clipped to the boundary, which would otherwise distort r.
    """
    if radius < 0:
        raise ValueError("Radius must be nonnegative")
    if radius == 0:
        return np.repeat(reference[None,:],n,axis=0)
    span = bounds[:,1]-bounds[:,0]
    lo = np.maximum(-1.0, (bounds[:,0]-reference)/(radius*span))
    hi = np.minimum(+1.0, (bounds[:,1]-reference)/(radius*span))
    directions = [(i,sgn) for i in range(6) for sgn in (-1.,1.)
                  if (lo[i] <= sgn <= hi[i])]
    if not directions:
        raise ValueError(f"No point exists at radius {radius} inside this box")
    result=np.empty((n,6),dtype=float)
    for j in range(n):
        z=rng.uniform(lo,hi)
        i,sgn=directions[int(rng.integers(len(directions)))]
        z[i]=sgn
        result[j]=reference+radius*span*z
    assert np.all(result>=bounds[:,0]-1e-12) and np.all(result<=bounds[:,1]+1e-12)
    return result


def corners(bounds:np.ndarray) -> np.ndarray:
    """All 2**6 = 64 vertices, for a finite stress test (not a proof)."""
    return np.array([[bounds[i,c[i]] for i in range(6)]
                     for c in itertools.product((0,1),repeat=6)],dtype=float)


def generate_laws(etas: np.ndarray,cfg: QuantileConfig,description:str) -> tuple[np.ndarray,np.ndarray]:
    """Return N x quantiles and N x bin probabilities, checking common grids."""
    q=np.empty((len(etas),cfg.n_quantiles),dtype=float)
    p=np.empty((len(etas),cfg.n_bins),dtype=float)
    for i,eta in enumerate(etas):
        law=heston_quantiles(eta,cfg)
        q[i]=law.quantiles
        p[i]=law.probabilities
        if (i+1)%100==0:
            print(f"  {description}: {i+1}/{len(etas)} states",flush=True)
    return q,p


def fit_reference_modes(train_quantiles: np.ndarray,reference_quantiles: np.ndarray,
                        u: np.ndarray,max_modes: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Reference-anchored (UN-CENTRED) weighted quantile SVD.

    D_i(u)=Q_i(u)-Q_ref(u). With trapezoidal weights w,
    D_w[i,j]=D_i(u_j)*sqrt(w_j). SVD D_w=U*S*Vt yields modes
        b_k(u_j)=Vt[k,j]/sqrt(w_j).
    Thus <b_k,b_l>_w=delta_kl and the reference uses ZERO coefficients.

    This is intentionally not the same as centred PCA from the earlier repo.
    Centring would optimally approximate deviations about the training mean,
    but could change the exact reference anchor.
    """
    if max_modes<1 or max_modes>min(train_quantiles.shape):
        raise ValueError("max_modes must be between 1 and min(n_train,n_quantiles)")
    weight=grid_weights(u)
    mat=(train_quantiles-reference_quantiles)*np.sqrt(weight)
    _,singular,vt=np.linalg.svd(mat,full_matrices=False)
    energy=singular**2
    fractions=energy[:max_modes]/energy.sum()
    modes=vt[:max_modes]/np.sqrt(weight)
    # Fix the irrelevant SVD sign for deterministic cross-platform output.
    for k,mode in enumerate(modes):
        imax=np.argmax(np.abs(mode))
        if mode[imax]<0:
            modes[k]*=-1
    return modes,fractions,singular


def project(reference_quantiles:np.ndarray, target_quantiles:np.ndarray,
            modes:np.ndarray,weights:np.ndarray,n_modes:int) -> tuple[np.ndarray,np.ndarray]:
    """Weighted LEAST-SQUARES fit of k reference-anchored modes.

    Requires knowledge of Q_target. This is an oracle projection, NOT a
    prediction from Heston parameters alone and NOT a quantum unitary.
    """
    if not 0 <= n_modes <= len(modes):
        raise ValueError(f"Number of modes must be in [0,{len(modes)}]")
    coeff=(target_quantiles-reference_quantiles)*weights @ modes[:n_modes].T
    qhat=reference_quantiles+coeff @ modes[:n_modes]
    return qhat,coeff


def fidelity(p:np.ndarray,q:np.ndarray)->float:
    """Squared overlap of positive-real amplitude encodings sqrt(p),sqrt(q)."""
    return float(np.dot(np.sqrt(p),np.sqrt(q))**2)


def evaluate_rank(quantiles:np.ndarray,probs:np.ndarray,reference_q:np.ndarray,
                  modes:np.ndarray,u:np.ndarray,edges:np.ndarray,
                  n_modes:int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return per-state (amplitude fidelity, truncated W2, monotonicity repairs)."""
    weights=grid_weights(u)
    qs,coeff=project(reference_q,quantiles,modes,weights,n_modes)
    fid=np.empty(len(quantiles));w2=np.empty(len(quantiles));repair=np.empty(len(quantiles),dtype=bool)
    for j in range(len(quantiles)):
        pp,corrected,flag=quantiles_to_probabilities(qs[j],u,edges)
        fid[j]=fidelity(probs[j],pp)
        w2[j]=np.sqrt(np.sum(weights*(quantiles[j]-corrected)**2))
        repair[j]=flag
    return fid,w2,repair


def summary_stats(vals:np.ndarray, w2:np.ndarray, repairs:np.ndarray) ->dict:
    return {"mean_fidelity":float(np.mean(vals)),
            "p05_fidelity":float(np.quantile(vals,.05)),
            "min_fidelity":float(np.min(vals)),
            "mean_w2":float(np.mean(w2)),
            "fraction_isotonic_repaired":float(np.mean(repairs))}


def empirical_reach(rows: list[dict], rank: int, threshold:float) -> float | None:
    """Largest scanned radius up to which *every preceding shell* passed.

    This is an observed radius-grid diagnostic, not a guaranteed valid ball.
    A missing value means even the reference (at r=0) failed the fidelity
    threshold, perhaps from quantile-to-bin discretisation error.
    """
    ordered=sorted((r for r in rows if r['n_modes']==rank),key=lambda x:x['radius'])
    last=None
    for r in ordered:
        if r['min_fidelity']<threshold:
            break
        last=float(r['radius'])
    return last


def write_csv(path:Path,rows:list[dict]) -> None:
    if not rows:
        return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(rows)


def parse_bounds(filename:Path|None) ->np.ndarray:
    if filename is None:
        return ORIGINAL_BOUNDS.copy()
    raw=json.loads(filename.read_text())
    if isinstance(raw,dict):
        raw=[raw[name] for name in PARAMETER_NAMES]
    return np.asarray(raw,dtype=float)


def make_plots(out:Path,rank_rows:list[dict],radius_rows:list[dict],n_ranks:int) ->None:
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    k=np.arange(1,n_ranks+1)
    axes[0].bar(k,[r['energy_fraction'] for r in rank_rows if r['n_modes']>0])
    axes[0].set(xlabel='Transport mode number',ylabel='Fraction of displacement energy',title='Reference-anchored quantile SVD')
    axes[1].plot([r['n_modes'] for r in rank_rows],[r['validation_min_fidelity'] for r in rank_rows],'.-',label='Worst validation')
    axes[1].plot([r['n_modes'] for r in rank_rows],[r['validation_p05_fidelity'] for r in rank_rows],'.-',label='5th percentile validation')
    axes[1].set(xlabel='Number of transport modes',ylabel='Amplitude fidelity',title='Fidelity against Heston probabilities')
    axes[1].legend()
    fig.tight_layout();fig.savefig(out/'mode_significance.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    ranks=sorted(set(int(row['n_modes']) for row in radius_rows))
    for rank in ranks:
        rr=sorted([row for row in radius_rows if row['n_modes']==rank],key=lambda x:x['radius'])
        axes[0].plot([x['radius'] for x in rr],[x['mean_fidelity'] for x in rr],'.-',label=f'{rank} modes')
        axes[1].semilogy([x['radius'] for x in rr],
                         [max(1e-12,1-x['min_fidelity']) for x in rr],'.-',label=f'{rank} modes')
    axes[0].set(xlabel='Normalised L∞ parameter radius',ylabel='Mean amplitude fidelity',title='Reference-to-target projection')
    axes[1].set(xlabel='Normalised L∞ parameter radius',ylabel='Worst sampled infidelity (1 - F)',title='Failure-sensitive distance profile')
    axes[1].legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(out/'fidelity_vs_radius.png',dpi=160);plt.close(fig)


def run(*,reference:np.ndarray,bounds:np.ndarray,cfg:QuantileConfig,
        out:Path, n_train:int=256,n_validation:int=96,n_test:int=128,
        max_modes:int=12,n_per_radius:int=32,
        radii:list[float]|None=None,threshold:float=.9999,
        criterion:str='worst',seed:int=2026, historical_dataset:Path|None=None) ->dict:
    """Fit only on TRAIN, select k using VALIDATION+corners, test once.

    The radius probe is diagnostic and is not used for mode selection. The
    corner set is included in selection: this deliberately asks for reliable
    performance across extremes. Neither finite corners nor heldout states
    certify uniform accuracy on a continuous 6D parameter box.
    """
    t0=time.perf_counter()
    reference=validate_eta(reference)
    bounds=validate_box(bounds,reference)
    cfg.validate()
    if radii is None:
        radii=[0.,.01,.02,.05,.1,.2,.35,.5]
    radii=sorted(set(float(r) for r in radii))
    if not radii or any(r<0 for r in radii) or n_per_radius<2:
        raise ValueError("Provide nonnegative radii and n_per_radius>=2")
    if not 0<threshold<=1 or criterion not in ('worst','p05'):
        raise ValueError("Invalid threshold or criterion")
    out.mkdir(parents=True,exist_ok=True)
    print('Computing fixed Heston reference quantiles...',flush=True)
    ref=heston_quantiles(reference,cfg)
    if historical_dataset is not None:
        # Reuse the 2100 states, 512-point quantiles, and 70/15/15 split
        # from the original repo, rather than recomputing new Heston models.
        print(f'Loading original optimal-transport dataset: {historical_dataset}',flush=True)
        with np.load(historical_dataset,allow_pickle=False) as z:
            required={'params','quantiles','probability_bins','u_grid','bin_edges',
                      'train_indices','validation_indices','test_indices'}
            if not required.issubset(z.files):
                raise ValueError(f'Historical dataset missing: {sorted(required-set(z.files))}')
            all_eta=z['params'];all_q=z['quantiles'];all_p=z['probability_bins']
            if all_eta.ndim!=2 or all_eta.shape[1]!=6 or \
               all_q.shape!=(len(all_eta),cfg.n_quantiles) or \
               all_p.shape!=(len(all_eta),cfg.n_bins):
                raise ValueError('Historical dataset incompatible with requested resolution')
            if not np.allclose(z['u_grid'],ref.u,rtol=0,atol=1e-13) or \
               not np.allclose(z['bin_edges'],ref.bin_edges,rtol=0,atol=1e-12):
                raise ValueError('Historical dataset has a different quantile/bin grid')
            if np.any(all_eta<bounds[:,0]-1e-12) or np.any(all_eta>bounds[:,1]+1e-12):
                raise ValueError('Historical dataset includes models outside chosen parameter box')
            indices=[np.asarray(z[k],dtype=int) for k in
                     ('train_indices','validation_indices','test_indices')]
            if any(np.any(idx<0) or np.any(idx>=len(all_eta)) for idx in indices) or \
               len(np.unique(np.concatenate(indices)))!=sum(map(len,indices)):
                raise ValueError('Historical train/validation/test indices overlap or are invalid')
        tr,va,te=indices
        train,val,test=all_eta[tr],all_eta[va],all_eta[te]
        train_q=all_q[tr];val_q,val_p=all_q[va],all_p[va]
        n_train,n_validation,n_test=len(train),len(val),len(test)
    else:
        train=sample_box(bounds,n_train,seed)
        val=sample_box(bounds,n_validation,seed+1)
        test=sample_box(bounds,n_test,seed+2)
        print('Building disjoint training and validation Heston laws...',flush=True)
        train_q,_=generate_laws(train,cfg,'train')
        val_q,val_p=generate_laws(val,cfg,'validation')
    corner_etas=corners(bounds)
    corners_q,corners_p=generate_laws(corner_etas,cfg,'corners')
    print('Fitting weighted reference-anchored transport basis...',flush=True)
    modes,energies,singular=fit_reference_modes(train_q,ref.quantiles,ref.u,max_modes)
    ranked=[];all_val={};all_corner={}
    for rank in range(max_modes+1):
        f_v,w_v,r_v=evaluate_rank(val_q,val_p,ref.quantiles,modes,ref.u,ref.bin_edges,rank)
        f_c,w_c,r_c=evaluate_rank(corners_q,corners_p,ref.quantiles,modes,ref.u,ref.bin_edges,rank)
        stats=summary_stats(f_v,w_v,r_v)
        cstats=summary_stats(f_c,w_c,r_c)
        row={'n_modes':rank,'energy_fraction':float(energies[rank-1]) if rank else 0.,
             'cumulative_energy':float(energies[:rank].sum()),
             **{'validation_'+k:v for k,v in stats.items()},
             **{'corners_'+k:v for k,v in cstats.items()}}
        ranked.append(row);all_val[rank]=f_v;all_corner[rank]=f_c
    for j,row in enumerate(ranked):
        if j:
            row['delta_validation_mean_fidelity']=row['validation_mean_fidelity']-ranked[j-1]['validation_mean_fidelity']
            row['delta_validation_min_fidelity']=row['validation_min_fidelity']-ranked[j-1]['validation_min_fidelity']
            row['delta_validation_mean_w2']=row['validation_mean_w2']-ranked[j-1]['validation_mean_w2']
        else:
            row['delta_validation_mean_fidelity']=0.
            row['delta_validation_min_fidelity']=0.
            row['delta_validation_mean_w2']=0.
    # Conservatively select on the combined (held-in-selection) validation + corners.
    score_key='min' if criterion=='worst' else 'p05'
    for j,row in enumerate(ranked):
        mixed=np.r_[all_val[j],all_corner[j]]
        row['selection_'+score_key+'_fidelity']=float(np.min(mixed) if criterion=='worst'
                                                 else np.quantile(mixed,.05))
    passing=[row for row in ranked if row['selection_'+score_key+'_fidelity']>=threshold]
    passed=bool(passing)
    chosen=int(min(row['n_modes'] for row in passing)) if passed else max_modes
    print(f'Selected {chosen} modes (criterion {criterion}, target {threshold}, passed={passed})')
    write_csv(out/'mode_significance.csv',ranked)

    # Holdout isn't consulted while selecting k. Evaluate every rank only for
    # subsequent post-selection scientific analysis (clearly labelled).
    print('Evaluating independent TEST models...',flush=True)
    if historical_dataset is not None:
        test_q,test_p=all_q[te],all_p[te]
    else:
        test_q,test_p=generate_laws(test,cfg,'test')
    test_rows=[]
    for rank in range(max_modes+1):
        f,w,r=evaluate_rank(test_q,test_p,ref.quantiles,modes,ref.u,ref.bin_edges,rank)
        test_rows.append({'n_modes':rank,**summary_stats(f,w,r)})
    write_csv(out/'test_by_rank.csv',test_rows)

    # Fixed radial probes from same reference; not used to select k.
    radius_rows=[];radius_sample_rows=[]
    rng=np.random.default_rng(seed+10)
    plot_ranks=sorted({0,1,2,3,4,5,chosen,max_modes} & set(range(max_modes+1)))
    for radius in radii:
        ray=sample_shell(bounds,reference,radius,n_per_radius,rng)
        ray_q,ray_p=generate_laws(ray,cfg,f'radius={radius:g}')
        for rank in plot_ranks:
            f,w,repair=evaluate_rank(ray_q,ray_p,ref.quantiles,modes,ref.u,ref.bin_edges,rank)
            radius_rows.append({'radius':radius,'n_modes':rank,**summary_stats(f,w,repair)})
            for i,(eta,fi,wi) in enumerate(zip(ray,f,w)):
                radius_sample_rows.append({'radius':radius,'n_modes':rank,'index':i,
                                           **dict(zip(PARAMETER_NAMES,map(float,eta))),
                                           'fidelity':float(fi),'w2':float(wi)})
    write_csv(out/'fidelity_vs_radius.csv',radius_rows)
    write_csv(out/'fidelity_vs_radius_samples.csv',radius_sample_rows)
    reach={str(rank):empirical_reach(radius_rows,rank,threshold) for rank in plot_ranks}
    (out/'empirical_reach.json').write_text(json.dumps({
        'threshold':threshold,'observed_consecutive_passing_radius_by_mode_count':reach,
        'interpretation':'Finite sampled shells only. Not a certified error bound over the ball.'},indent=2))
    make_plots(out,ranked,radius_rows,max_modes)
    # Plot the actual *functions* b_k(u), not only their variance weights.
    fig,ax=plt.subplots(figsize=(8,5))
    for i in range(min(5,len(modes))):
        scale=max(float(np.max(np.abs(modes[i]))),1e-12)
        ax.plot(ref.u,modes[i]/scale,label=f'mode {i+1}')
    ax.set(xlabel='Quantile level u',ylabel='Mode shape (each individually rescaled)',
           title='Reference-anchored transport basis functions')
    ax.legend()
    fig.tight_layout();fig.savefig(out/'mode_functions.png',dpi=160);plt.close(fig)

    # Save one explicitly constructed reference -> target transport per radius.
    # A 1D monotone transport is given by the paired quantile samples
    # (x_source=Q_ref(u), x_target=Q_hat(u)). The target quantile needs
    # monotonic correction where the low-rank approximation folds.
    examples_eta=[];examples_q=[];examples_disp=[];examples_radius=[]
    rng_examples=np.random.default_rng(seed+999)
    example_rank=min(5,max_modes)
    for r in radii:
        eta=sample_shell(bounds,reference,r,1,rng_examples)[0]
        target=heston_quantiles(eta,cfg)
        pred,_=project(ref.quantiles,target.quantiles,modes,grid_weights(ref.u),example_rank)
        _,mono,_=quantiles_to_probabilities(pred,ref.u,ref.bin_edges)
        examples_eta.append(eta);examples_q.append(mono)
        examples_disp.append(mono-ref.quantiles);examples_radius.append(r)
    np.savez_compressed(out/'transport_examples.npz',
                        radius=np.array(examples_radius),eta=np.stack(examples_eta),
                        u_grid=ref.u,reference_quantiles=ref.quantiles,
                        fitted_quantiles=np.stack(examples_q),
                        displacement=np.stack(examples_disp),n_modes=np.array(example_rank))

    np.savez_compressed(out/'transport_basis.npz',
        schema_version=np.array(SCHEMA_VERSION),
        reference_eta=reference,reference_quantiles=ref.quantiles,
        reference_probabilities=ref.probabilities,
        bounds=bounds,u_grid=ref.u,bin_edges=ref.bin_edges,
        weights=grid_weights(ref.u),modes=modes,
        singular_values=singular,energy_fraction=energies,
        selected_modes=np.array(chosen),
        s0=np.array(cfg.s0),maturity=np.array(cfg.maturity),
        n_fft=np.array(cfg.n_fft),n_quantiles=np.array(cfg.n_quantiles),
        n_bins=np.array(cfg.n_bins),x_half_width=np.array(cfg.x_half_width),
        min_retained_mass=np.array(cfg.min_retained_mass),
        max_negative_mass=np.array(cfg.max_negative_mass))
    summary={'schema_version':SCHEMA_VERSION,
             'scope':'classical oracle weighted quantile transport, NOT a quantum circuit',
             'parameters':list(PARAMETER_NAMES),'reference_eta':reference.tolist(),
             'bounds':bounds.tolist(),'config':asdict(cfg),
             'seed':seed,'n_train':n_train,'n_validation':n_validation,
             'n_test':n_test,'n_corners':len(corner_etas),
             'selection':{'criterion':criterion,'threshold':threshold,
                          'passed':passed,'selected_modes':chosen,
                          'failure_message':None if passed else
                            'No candidate rank met requested threshold: max_modes is a fallback, not a guarantee.'},
             'five_mode_validation':ranked[5] if max_modes>=5 else None,
             'selected_mode_validation':ranked[chosen],
             'selected_mode_test':test_rows[chosen],
             'radii':radii,'radius_samples_per_shell':n_per_radius,
             'empirical_reach_by_mode_count':reach,
             'historical_dataset':str(historical_dataset) if historical_dataset is not None else None,
             'elapsed_seconds':round(time.perf_counter()-t0,2),
             'caveats':['Basis fitted to true Heston quantiles: new-model projection requires true target quantiles.',
                        'Uncentred SVD of displacements from exact reference, not centred PCA.',
                        'Expected amplitude fidelity is from classical binned distributions, not a quantum gate implementation.',
                        'Corner and random/radial checks do not certify the entire continuous parameter box.',
                        'Quantile-to-bin interpolation and isotonic correction can impose a discretisation floor.',
                        'Only terminal log-price S0=100, maturity T=1 by default; not full price paths.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'artifact':str(out/'transport_basis.npz'),
                      'selected_modes':chosen,'threshold_passed':passed,
                      'five_mode_validation':summary['five_mode_validation'],
                      'selected_mode_test':summary['selected_mode_test']},indent=2),flush=True)
    return summary


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('transport_modes/results/main'))
    ap.add_argument('--reference',type=float,nargs=6,default=None,
                    help='Six Heston parameters; default original range midpoints')
    ap.add_argument('--historical-dataset',type=Path,default=None,
                    help='Optional optimal-transport/results/dataset.npz (original 2100 samples and splits)')
    ap.add_argument('--bounds-json',type=Path,default=None,
                    help='JSON 6x2 bounds array or dict {mu:[low,high],...}')
    ap.add_argument('--n-train',type=int,default=256)
    ap.add_argument('--n-validation',type=int,default=96)
    ap.add_argument('--n-test',type=int,default=128)
    ap.add_argument('--max-modes',type=int,default=12)
    ap.add_argument('--n-per-radius',type=int,default=32)
    ap.add_argument('--radii',type=float,nargs='+',default=[0,.01,.02,.05,.10,.20,.35,.50])
    ap.add_argument('--threshold',type=float,default=.9999)
    ap.add_argument('--criterion',choices=['worst','p05'],default='worst')
    ap.add_argument('--n-fft',type=int,default=4096)
    ap.add_argument('--n-quantiles',type=int,default=512)
    ap.add_argument('--n-bins',type=int,default=256)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--quick',action='store_true',help='Small smoke test, NOT research-grade')
    a=ap.parse_args()
    bounds=parse_bounds(a.bounds_json)
    reference=np.asarray(a.reference if a.reference is not None else bounds.mean(axis=1))
    if a.quick:
        a.n_train,a.n_validation,a.n_test=32,12,12
        a.n_per_radius=6
        a.n_fft=1024
        a.n_quantiles=128
        a.max_modes=min(a.max_modes,6)
        a.radii=[0.,.05,.2,.5]
    cfg=QuantileConfig(n_fft=a.n_fft,n_quantiles=a.n_quantiles,n_bins=a.n_bins)
    run(reference=reference,bounds=bounds,cfg=cfg,out=a.out,
        n_train=a.n_train,n_validation=a.n_validation,n_test=a.n_test,
        max_modes=a.max_modes,n_per_radius=a.n_per_radius,
        radii=a.radii,threshold=a.threshold,criterion=a.criterion,seed=a.seed,
        historical_dataset=a.historical_dataset)


if __name__=='__main__':
    main()