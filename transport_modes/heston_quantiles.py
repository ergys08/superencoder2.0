"""Heston terminal *log-price* quantiles on one consistent numerical grid.

A small adaptation of the original ergys08/optimal-transport ``src/heston.py``
workflow.  Uses the repository's HestonModel.characteristic_function rather than
maintaining a second stochastic-volatility formula.

Important: a quantile function is CLASSICAL data. This module does not prepare
or transport a quantum state. ``sqrt(bin_probabilities)`` is the target amplitude
vector used for an ideal, noiseless 8-qubit fidelity calculation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid

# Allow direct invocation: python transport_modes/heston_quantiles.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from heston import HestonModel, HestonParameters  # noqa: E402

PARAMETER_NAMES = ("mu", "v0", "kappa", "theta", "xi", "rho")
# Exactly the ranges used in the original optimal-transport experiment.
ORIGINAL_BOUNDS = np.asarray([
    [-0.03, 0.12], [0.01, 0.10], [0.8, 4.0],
    [0.02, 0.10], [0.15, 0.75], [-0.9, -0.1],
], dtype=np.float64)


@dataclass(frozen=True)
class QuantileConfig:
    s0: float = 100.0
    maturity: float = 1.0
    n_fft: int = 4096
    n_quantiles: int = 512
    n_bins: int = 256
    x_half_width: float = 2.2
    min_retained_mass: float = 0.999
    max_negative_mass: float = 0.01

    def validate(self) -> None:
        if self.s0 <= 0 or self.maturity <= 0 or self.x_half_width <= 0:
            raise ValueError("s0, maturity and x_half_width must be positive")
        if self.n_fft < 512 or self.n_fft & (self.n_fft - 1):
            raise ValueError("n_fft must be a power of two >= 512")
        if not 16 <= self.n_quantiles <= 8192 or not 2 <= self.n_bins <= self.n_fft:
            raise ValueError("Invalid n_quantiles or n_bins")
        if self.n_bins & (self.n_bins - 1):
            raise ValueError("n_bins must be a power of two")
        if not 0 < self.min_retained_mass <= 1 or self.max_negative_mass < 0:
            raise ValueError("Invalid mass tolerances")


@dataclass
class QuantileLaw:
    parameters: np.ndarray  # six Heston parameters
    u: np.ndarray  # tail-dense Chebyshev grid, strictly in (0,1)
    quantiles: np.ndarray  # log-price quantile Q_eta(u)
    bin_edges: np.ndarray  # fixed log-price grid; same for ALL models in a study
    probabilities: np.ndarray  # probability in each bin, normalized to 1
    retained_mass: float  # mass within the FFT integration window
    negative_fft_mass: float  # amount clipped from FFT numerical oscillations

    @property
    def amplitudes(self) -> np.ndarray:
        return np.sqrt(self.probabilities)


def validate_eta(eta: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(eta, dtype=np.float64)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ValueError("eta must contain six finite parameters in order mu,v0,kappa,theta,xi,rho")
    # HestonParameters will do the remaining checks (including rho).
    HestonParameters(**dict(zip(PARAMETER_NAMES, arr)))
    return arr


def u_grid(n_quantiles: int) -> np.ndarray:
    j = np.arange(n_quantiles, dtype=np.float64)
    return (1.0 - np.cos(np.pi * (j + 0.5) / n_quantiles)) / 2.0


def grid_weights(u: np.ndarray) -> np.ndarray:
    """Trapezoid quadrature approximation to the W2 quantile inner product."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or len(u) < 2 or np.any(np.diff(u) <= 0):
        raise ValueError("u must be a strictly increasing 1D grid")
    w = np.empty_like(u)
    w[0], w[-1] = 0.5 * (u[1] - u[0]), 0.5 * (u[-1] - u[-2])
    w[1:-1] = 0.5 * (u[2:] - u[:-2])
    return w


def bin_edges_for(cfg: QuantileConfig) -> np.ndarray:
    """Same bin convention as optimal-transport/src/heston.py: x[0] to x[-1]."""
    dx = 2.0 * cfg.x_half_width / cfg.n_fft
    x0 = np.log(cfg.s0) - cfg.x_half_width
    return np.linspace(x0, x0 + dx * (cfg.n_fft - 1), cfg.n_bins + 1)


def heston_quantiles(eta: np.ndarray | list[float] | HestonParameters | HestonModel,
                     cfg: QuantileConfig = QuantileConfig()) -> QuantileLaw:
    """Construct a normalized Heston terminal law and its quantile function.

    Uses the *original* numerical scheme: periodic FFT inversion of log-price
    CF -> remove small negative FFT artifacts -> trapezoid CDF -> quantile
    interpolation -> common 256-bin histogram. Does not simulate paths.

    The original x-grid does not include the right window endpoint; check
    retained mass and numerical negative mass before trusting the output.
    """
    cfg.validate()
    if isinstance(eta, HestonModel):
        model = eta
        p = model.params
        if not np.isclose([p.s0, p.maturity], [cfg.s0, cfg.maturity], rtol=0, atol=1e-12).all():
            raise ValueError("HestonModel s0/maturity disagree with numerical config")
        vec = np.array([getattr(p, k) for k in PARAMETER_NAMES], dtype=np.float64)
    elif isinstance(eta, HestonParameters):
        vec = np.array([getattr(eta, k) for k in PARAMETER_NAMES], dtype=np.float64)
        model = HestonModel(eta)
        if not np.isclose([eta.s0, eta.maturity], [cfg.s0, cfg.maturity], rtol=0, atol=1e-12).all():
            raise ValueError("Parameter s0/maturity disagree with numerical config")
    else:
        vec = validate_eta(eta)
        pars = HestonParameters(s0=cfg.s0, maturity=cfg.maturity,
                                **dict(zip(PARAMETER_NAMES, map(float, vec))))
        model = HestonModel(pars)

    dx = 2 * cfg.x_half_width / cfg.n_fft
    x0 = np.log(cfg.s0) - cfg.x_half_width
    x = x0 + dx * np.arange(cfg.n_fft)
    omega = 2 * np.pi * np.fft.fftfreq(cfg.n_fft, dx)
    cf = model.characteristic_function(omega)
    raw_pdf = (np.fft.fft(cf * np.exp(-1j * omega * x0)).real
               / (cfg.n_fft * dx))
    negative_mass = float(dx * np.maximum(-raw_pdf, 0).sum())
    density = np.maximum(raw_pdf, 0)
    cdf_raw = np.r_[0.0, cumulative_trapezoid(density, x)]
    mass = float(cdf_raw[-1])
    if not np.isfinite(mass) or mass < cfg.min_retained_mass:
        raise ValueError(f"FFT window retained only mass {mass:.8f}; enlarge x_half_width")
    if negative_mass > cfg.max_negative_mass:
        raise ValueError(f"FFT inversion clipped negative mass {negative_mass:.8f}; refine grid")
    cdf = cdf_raw / mass
    u = u_grid(cfg.n_quantiles)
    q = np.interp(u, cdf, x)
    edges = bin_edges_for(cfg)
    p_edges = np.interp(edges, x, cdf)
    probabilities = np.maximum(np.diff(p_edges), 0.0)
    total = float(probabilities.sum())
    if total <= 0:
        raise ArithmeticError("Invalid bin probabilities")
    probabilities /= total
    if not np.all(np.diff(q) >= 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ArithmeticError("Invalid quantile/CDF output")
    return QuantileLaw(vec, u, q, edges, probabilities, mass, negative_mass)


def quantiles_to_probabilities(q: np.ndarray, u: np.ndarray,
                               edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Turn reconstructed quantiles into a valid binned probability vector.

    Quantile PCA can violate monotonicity. Weighted isotonic regression repairs
    this; because it changes the answer, return ``corrected`` explicitly.
    Extrapolated u-tail masses follow the original optimal-transport convention.
    """
    from sklearn.isotonic import IsotonicRegression
    q = np.asarray(q, dtype=float).copy()
    u = np.asarray(u, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if q.shape != u.shape or edges.ndim != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("Quantiles/grid/bin edges have incompatible shapes")
    corrected = bool(np.any(np.diff(q) <= 0))
    if corrected:
        q = IsotonicRegression(increasing=True).fit_transform(u, q,
                                              sample_weight=grid_weights(u))
    q = np.maximum.accumulate(q + 1e-10 * np.arange(len(q)))
    knots_x = np.r_[edges[0], q, edges[-1]]
    knots_u = np.r_[0.0, u, 1.0]
    knots_x = np.maximum.accumulate(np.clip(knots_x, edges[0], edges[-1]))
    knots_x += 1e-11 * np.arange(len(knots_x))
    cdf_edges = np.interp(edges, knots_x, knots_u, left=0., right=1.)
    probs = np.maximum(np.diff(cdf_edges), 0)
    if probs.sum() <= 0:
        raise ValueError("Quantile inversion produced zero probability mass")
    probs /= probs.sum()
    return probs, q, corrected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eta", nargs=6, type=float, required=True,
                   metavar=tuple(k.upper() for k in PARAMETER_NAMES))
    p.add_argument("--out", type=Path, default=Path("transport_modes/results/example_quantiles.npz"))
    p.add_argument("--n-fft", type=int, default=4096)
    args = p.parse_args()
    cfg = QuantileConfig(n_fft=args.n_fft)
    law = heston_quantiles(args.eta, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, eta=law.parameters, u=law.u,
                        quantiles=law.quantiles, bin_edges=law.bin_edges,
                        probabilities=law.probabilities)
    print(json.dumps({"output": str(args.out), "quantiles": len(law.u),
                      "bins": len(law.probabilities),
                      "retained_mass": law.retained_mass,
                      "negative_fft_mass": law.negative_fft_mass,
                      "config": asdict(cfg)}, indent=2))


if __name__ == "__main__":
    main()