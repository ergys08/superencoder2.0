"""Robustness tests over the WHOLE prescribed six-parameter Heston box.

The goal is strong transport *from one fixed reference* to any admissible target.
We cannot certify a minimum over an uncountable domain with finite sampling:
these tests provide falsification opportunities and an empirical worst case.

Three complementary probes:
    1. All 2**6 = 64 corners (deterministic boundary stress test).
    2. Seeded Latin-hypercube interior sample.
    3. Seeded differential-evolution search for minimum fidelity.

The same public 64 corners appear in finalist selection and final audit. The
random/adversarial points use independent seeds in the two phases. No held-out
state is passed into this module for architecture selection.
"""
from __future__ import annotations

from itertools import product
from dataclasses import dataclass
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from config import BOUNDS, Settings
from data import amplitude
from learner import predict


def corners() -> np.ndarray:
    """Exactly the 64 vertices of the closed six-dimensional parameter box."""
    return np.array([[BOUNDS[j, bit] for j, bit in enumerate(bits)]
                     for bits in product((0, 1), repeat=6)], dtype=float)


def interior(n: int, seed: int) -> np.ndarray:
    """Independent seeded space-filling sample, never used to train the NN."""
    if n < 0:
        raise ValueError('Number of random points cannot be negative')
    if not n:
        return np.empty((0, 6), dtype=float)
    return qmc.scale(qmc.LatinHypercube(d=6, seed=seed).random(n),
                     BOUNDS[:, 0], BOUNDS[:, 1])


def fidelity_at(model, reference: np.ndarray, eta: np.ndarray,
                settings: Settings, bin_edges: np.ndarray | None = None) -> float:
    """Evaluate exact ideal-state transport fidelity at ONE parameter vector.

    The target is recomputed with the SAME discretisation as the reference.
    A new target-specific optimisation is never performed.
    """
    x = np.asarray(eta, dtype=float)
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError('Expected finite six-dimensional Heston parameters')
    if np.any(x < BOUNDS[:, 0]) or np.any(x > BOUNDS[:, 1]):
        raise ValueError('Parameters outside predefined domain')
    target = amplitude(x, settings.n_qubits, settings.n_fft, bin_edges)
    pred, _ = predict(model, x.reshape(1, -1), reference)
    value = float(abs(np.vdot(target, pred[0])) ** 2)
    if not np.isfinite(value):
        raise ArithmeticError('Nonfinite transport fidelity')
    return float(np.clip(value, 0.0, 1.0))


def evaluate_points(model, reference: np.ndarray, points: np.ndarray,
                    settings: Settings, bin_edges: np.ndarray | None = None) -> np.ndarray:
    """Batch NN inference on exact Heston states at the supplied points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError('Expected (n, 6) parameter array')
    if len(points) == 0:
        return np.empty(0, dtype=float)
    targets = np.stack([amplitude(eta, settings.n_qubits, settings.n_fft, bin_edges)
                        for eta in points])
    predictions, _ = predict(model, points, reference)
    values = np.square(np.abs(np.sum(targets * np.conj(predictions), axis=1)))
    if not np.all(np.isfinite(values)):
        raise ArithmeticError('Nonfinite transport fidelity in domain evaluation')
    return np.clip(values.real, 0.0, 1.0)


@dataclass
class RobustnessReport:
    """All values are empirical; `observed_min` is NOT a certified minimum."""
    observed_min_fidelity: float
    observed_p05_fidelity: float
    observed_mean_fidelity: float
    corner_min_fidelity: float
    interior_min_fidelity: float | None
    adversarial_min_fidelity: float | None
    worst_eta: list[float]
    worst_source: str
    n_corners: int
    n_interior: int
    adversarial_nfev: int
    pass_observed_min_threshold: bool
    # Full arrays make each experiment independently auditable.
    points: np.ndarray
    fidelities: np.ndarray
    labels: list[str]

    def summary(self) -> dict:
        """Serializable metrics; explicit scope and no claim of proof."""
        return {
            'observed_min_fidelity': self.observed_min_fidelity,
            'observed_p05_fidelity': self.observed_p05_fidelity,
            'observed_mean_fidelity': self.observed_mean_fidelity,
            'corner_min_fidelity': self.corner_min_fidelity,
            'interior_min_fidelity': self.interior_min_fidelity,
            'adversarial_min_fidelity': self.adversarial_min_fidelity,
            'worst_eta': self.worst_eta,
            'worst_source': self.worst_source,
            'n_corners': self.n_corners,
            'n_interior': self.n_interior,
            'adversarial_nfev': self.adversarial_nfev,
            'pass_observed_min_threshold': self.pass_observed_min_threshold,
            'note': 'Sampled / numerically searched worst case, NOT a global certificate.',
        }


def audit(model, reference: np.ndarray, settings: Settings, *, seed: int,
          n_interior: int, adversarial_maxiter: int,
          adversarial_popsize: int, include_corners: bool = True,
          target_fidelity: float | None = None,
          bin_edges: np.ndarray | None = None) -> RobustnessReport:
    """Evaluate a frozen parameter->angle NN throughout the permitted domain.

    `seed` must differ between validation selection and the final audit.
    Differential evolution searches for a low fidelity point; it does not
    change circuit angles, NN weights, circuit structure, or the reference.
    """
    if adversarial_maxiter < 0 or adversarial_popsize < 0:
        raise ValueError('Invalid adversarial search budget')
    if n_interior < 0:
        raise ValueError('Invalid random sample size')
    if target_fidelity is None:
        target_fidelity = settings.target_fidelity

    corner_pts = corners() if include_corners else np.empty((0, 6))
    interior_pts = interior(n_interior, seed)
    pts = np.vstack([corner_pts, interior_pts])
    if not len(pts):
        raise ValueError('At least one robustness check is required')
    vals = evaluate_points(model, reference, pts, settings, bin_edges)
    labels = ['corner'] * len(corner_pts) + ['interior'] * len(interior_pts)
    corner_min = float(vals[:len(corner_pts)].min()) if len(corner_pts) else float('nan')
    interior_min = float(vals[len(corner_pts):].min()) if len(interior_pts) else None

    adv_min = None
    nfev = 0
    if adversarial_maxiter > 0 and adversarial_popsize > 0:
        result = differential_evolution(
            lambda x: fidelity_at(model, reference, x, settings, bin_edges),
            bounds=[tuple(b) for b in BOUNDS],
            seed=seed + 202,
            maxiter=adversarial_maxiter,
            popsize=adversarial_popsize,
            polish=False,
            updating='immediate',
            workers=1,
        )
        adv_min = float(result.fun)
        nfev = int(result.nfev)
        pts = np.vstack([pts, result.x])
        vals = np.append(vals, adv_min)
        labels.append('adversarial')

    idx = int(np.argmin(vals))
    return RobustnessReport(
        observed_min_fidelity=float(vals.min()),
        observed_p05_fidelity=float(np.quantile(vals, .05)),
        observed_mean_fidelity=float(vals.mean()),
        corner_min_fidelity=corner_min,
        interior_min_fidelity=interior_min,
        adversarial_min_fidelity=adv_min,
        worst_eta=pts[idx].tolist(),
        worst_source=labels[idx],
        n_corners=len(corner_pts),
        n_interior=len(interior_pts),
        adversarial_nfev=nfev,
        pass_observed_min_threshold=bool(vals.min() >= target_fidelity),
        points=pts,
        fidelities=vals,
        labels=labels,
    )