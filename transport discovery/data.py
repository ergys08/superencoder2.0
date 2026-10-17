"""Construct fixed reference and disjoint Heston distribution datasets.

No Monte Carlo path sampling: we Fourier-invert the analytic terminal
log-price characteristic function using the repo's existing HestonModel.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from heston import HestonModel, HestonParameters  # noqa: E402
from config import BOUNDS, PARAMETER_NAMES, REFERENCE, Settings


@dataclass
class StateDataset:
    eta_train: np.ndarray
    psi_train: np.ndarray
    eta_val: np.ndarray
    psi_val: np.ndarray
    eta_test: np.ndarray
    psi_test: np.ndarray
    psi_ref: np.ndarray
    eta_ref: np.ndarray


def amplitude(eta: np.ndarray, n_qubits: int, n_fft: int) -> np.ndarray:
    """Return sqrt(discrete bin masses) for the fixed S0=100, T=1 law."""
    param = HestonParameters(s0=100., maturity=1., **dict(zip(PARAMETER_NAMES, map(float, eta))))
    return HestonModel(param).amplitude_state(n_qubits=n_qubits, n_fft=n_fft)


def _stratified_lhs(total: int, seed: int) -> np.ndarray:
    u = qmc.LatinHypercube(d=6, seed=seed).random(n=total)
    return qmc.scale(u, BOUNDS[:, 0], BOUNDS[:, 1])


def create_dataset(s: Settings, external_npz: str | None = None) -> StateDataset:
    """Load original PCA dataset if given; otherwise generate reproducible new states.

    The optional external NPZ must be results/dataset.npz from optimal-transport,
    which includes train_indices, validation_indices and test_indices.
    Its original split is preserved to avoid information leakage.
    """
    if external_npz:
        z = np.load(external_npz)
        needed = ('params','probability_bins','train_indices',
                  'validation_indices','test_indices')
        missing = [key for key in needed if key not in z]
        if missing:
            raise ValueError(f'Dataset missing fields: {missing}')
        params = np.asarray(z['params'], dtype=np.float64)
        probs = np.asarray(z['probability_bins'], dtype=np.float64)
        if params.shape != (len(probs), 6) or probs.shape[1] != 2**s.n_qubits:
            raise ValueError('Provided dataset has incompatible shapes')
        if (params < BOUNDS[:, 0]-1e-10).any() or (params > BOUNDS[:, 1]+1e-10).any():
            raise ValueError('Dataset violates original Heston parameter bounds')
        if not np.all(np.isfinite(probs)) or np.any(probs < -1e-12):
            raise ValueError('Invalid probabilities in saved dataset')
        psi = np.sqrt(np.maximum(probs, 0) / np.maximum(probs.sum(axis=1, keepdims=True), 1e-300))
        tr, va, te = [np.asarray(z[k], dtype=int) for k in ('train_indices','validation_indices','test_indices')]
        if len(set(tr.tolist()+va.tolist()+te.tolist())) != len(tr)+len(va)+len(te):
            raise ValueError('Overlapping state-level splits')
        # Fixed reference must be computed with *same* binning convention as NPZ.
        # Original NPZ binning differs slightly from heston.py: use interpolation
        # of the original quantile function instead of mixing conventions.
        # The same reference is an independent theoretical Heston law, rebinned
        # below using the original dataset's fixed edges.
        from scipy.interpolate import interp1d
        if 'bin_edges' not in z:
            raise ValueError('Original dataset must contain bin_edges')
        edges = np.asarray(z['bin_edges'])
        dist = HestonModel(HestonParameters(**dict(zip(PARAMETER_NAMES, map(float,REFERENCE))))).terminal_distribution(
            n_qubits=s.n_qubits,n_fft=s.n_fft)
        source_edges = dist.log_price_edges
        ref_cdf = np.r_[0., np.cumsum(dist.probabilities)]
        ref_probs = np.maximum(np.diff(np.interp(edges, source_edges, ref_cdf, left=0., right=1.)), 0.)
        if ref_probs.sum() < .999:
            raise ValueError('Reference distribution falls outside supplied bins')
        psi_ref = np.sqrt(ref_probs / ref_probs.sum())
        return StateDataset(params[tr], psi[tr],params[va],psi[va],params[te],psi[te],psi_ref,REFERENCE.copy())

    all_eta = _stratified_lhs(s.n_train + s.n_validation + s.n_test, s.seed)
    all_psi = np.stack([amplitude(eta, s.n_qubits, s.n_fft) for eta in all_eta])
    if np.min(np.linalg.norm(all_psi, axis=1)) < 0.999999:
        raise ArithmeticError('Heston states failed normalization test')
    rng = np.random.default_rng(s.seed + 19)
    shuffled = rng.permutation(len(all_eta))
    tr = shuffled[:s.n_train]
    va = shuffled[s.n_train:s.n_train + s.n_validation]
    te = shuffled[s.n_train + s.n_validation:]
    return StateDataset(all_eta[tr],all_psi[tr],all_eta[va],all_psi[va],
                        all_eta[te],all_psi[te],amplitude(REFERENCE,s.n_qubits,s.n_fft),REFERENCE.copy())


def save_dataset_metadata(s: Settings, out: Path, external: str | None) -> None:
    import json
    out.mkdir(parents=True, exist_ok=True)
    (out/'domain.json').write_text(json.dumps({
        'parameter_names': PARAMETER_NAMES,
        'bounds': BOUNDS.tolist(),
        'reference_parameters': REFERENCE.tolist(),
        's0':100., 'maturity':1.,
        'source': external or 'fresh-seeded-Latin-hypercube',
        'seed':s.seed,
        'n_qubits':s.n_qubits, 'n_fft':s.n_fft,
        'note':'FFT inversion and finite binning are approximate; no Monte Carlo.'
    }, indent=2))
