"""Reusable Heston stochastic-volatility model for SuperEncoder 2.0."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class HestonParameters:
    """SDE: dS=mu*S*dt+sqrt(v)*S*dW_s; dv=kappa*(theta-v)*dt+xi*sqrt(v)*dW_v."""

    s0: float = 100.0
    mu: float = 0.05
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.30
    rho: float = -0.70
    maturity: float = 1.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(list(self.__dict__.values()))):
            raise ValueError("Every Heston parameter must be finite")
        if self.s0 <= 0 or self.maturity <= 0:
            raise ValueError("s0 and maturity must be strictly positive")
        if self.v0 < 0 or self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
            raise ValueError("Require v0>=0 and kappa,theta,xi>0")
        if not -1 < self.rho < 1:
            raise ValueError("rho must lie strictly between -1 and 1")

    @property
    def feller_condition(self) -> bool:
        return 2.0 * self.kappa * self.theta >= self.xi**2

    def updated(self, **changes: float) -> "HestonParameters":
        return replace(self, **changes)


@dataclass(frozen=True)
class HestonDistribution:
    """Discrete terminal price law, with bins uniform in log-price."""
    log_price_edges: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    @property
    def price_edges(self) -> NDArray[np.float64]:
        return np.exp(self.log_price_edges)

    @property
    def price_midpoints(self) -> NDArray[np.float64]:
        return np.exp(0.5 * (self.log_price_edges[1:] + self.log_price_edges[:-1]))

    @property
    def amplitudes(self) -> NDArray[np.float64]:
        """Positive amplitude-encoding target state, normalized to unit norm."""
        return np.sqrt(self.probabilities)

    @property
    def n_qubits(self) -> int:
        n = len(self.probabilities)
        if n <= 0 or n & (n - 1):
            raise ValueError("Number of probabilities must be a power of two")
        return n.bit_length() - 1


class HestonModel:
    """A parameterised Heston model reusable across simulation and learning scripts."""

    def __init__(self, params: HestonParameters) -> None:
        if not isinstance(params, HestonParameters):
            raise TypeError("params must be HestonParameters")
        self.params = params

    def with_parameters(self, **changes: float) -> "HestonModel":
        """Return a new model without modifying this one."""
        return HestonModel(self.params.updated(**changes))

    def characteristic_function(self, u: ArrayLike) -> NDArray[np.complex128]:
        """Compute E[exp(i*u*log(S_T))] using the little-Heston-trap form."""
        p = self.params
        u = np.asarray(u, dtype=np.complex128)
        iu = 1j * u
        b = p.kappa - p.rho * p.xi * iu
        d = np.sqrt(b * b + p.xi**2 * (u * u + iu))
        g = (b - d) / (b + d)
        e = np.exp(-d * p.maturity)
        c = (p.kappa * p.theta / p.xi**2) * (
            (b - d) * p.maturity - 2 * np.log((1 - g * e) / (1 - g))
        )
        big_d = ((b - d) / p.xi**2) * (1 - e) / (1 - g * e)
        return np.exp(iu * (np.log(p.s0) + p.mu * p.maturity) + c + p.v0 * big_d)

    def terminal_distribution(
        self,
        n_qubits: int = 8,
        *,
        n_fft: int = 8192,
        log_price_half_width: float = 2.2,
        boundary_tolerance: float = 1e-3,
    ) -> HestonDistribution:
        """FFT-invert Heston terminal log-price CF into 2**n_qubits bins.

        The inversion uses a periodic log-price window. Increase width and FFT
        resolution when required. Edge-mass checking detects obvious aliasing.
        """
        if not isinstance(n_qubits, int) or not 1 <= n_qubits <= 20:
            raise ValueError("n_qubits must be an integer between 1 and 20")
        bins = 1 << n_qubits
        if n_fft < bins or (n_fft & (n_fft - 1)) or n_fft % bins:
            raise ValueError("n_fft must be a power of two divisible by 2**n_qubits")
        if log_price_half_width <= 0 or boundary_tolerance < 0:
            raise ValueError("Invalid window width or boundary tolerance")
        dx = 2.0 * log_price_half_width / n_fft
        x0 = np.log(self.params.s0) - log_price_half_width
        frequencies = 2 * np.pi * np.fft.fftfreq(n_fft, dx)
        transform = self.characteristic_function(frequencies) * np.exp(
            -1j * frequencies * x0
        )
        density = np.real(np.fft.fft(transform)) / (n_fft * dx)
        density = np.maximum(density, 0)
        masses = density * dx
        total = masses.sum()
        if not np.isfinite(total) or total <= 0:
            raise ArithmeticError("Fourier inversion produced an invalid distribution")
        masses /= total
        tail_cells = max(1, n_fft // 100)
        edge_mass = masses[:tail_cells].sum() + masses[-tail_cells:].sum()
        if edge_mass > boundary_tolerance:
            raise ValueError(
                f"FFT window may be too narrow (edge mass={edge_mass:.3g}); "
                "increase log_price_half_width or n_fft"
            )
        probabilities = masses.reshape(bins, n_fft // bins).sum(axis=1)
        probabilities /= probabilities.sum()
        edges = x0 + dx * np.arange(bins + 1) * (n_fft // bins)
        return HestonDistribution(edges, probabilities)

    def amplitude_state(self, n_qubits: int = 8, **kwargs: object) -> NDArray[np.float64]:
        """Return a normalised 2**n_qubits amplitude vector."""
        return self.terminal_distribution(n_qubits, **kwargs).amplitudes

    def simulate_paths(
        self, n_paths: int = 1000, n_steps: int = 252, *, seed: Optional[int] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Full-truncation Euler simulation: return (prices, variance) paths."""
        if n_paths < 1 or n_steps < 1:
            raise ValueError("n_paths and n_steps must be positive")
        p = self.params
        dt = p.maturity / n_steps
        rng = np.random.default_rng(seed)
        log_s = np.empty((n_paths, n_steps + 1))
        variance = np.empty_like(log_s)
        log_s[:, 0] = np.log(p.s0)
        variance[:, 0] = p.v0
        for t in range(n_steps):
            z_s, z_independent = rng.standard_normal((2, n_paths))
            z_v = p.rho * z_s + np.sqrt(1 - p.rho**2) * z_independent
            vpos = np.maximum(variance[:, t], 0)
            log_s[:, t + 1] = (
                log_s[:, t] + (p.mu - 0.5 * vpos) * dt
                + np.sqrt(vpos * dt) * z_s
            )
            variance[:, t + 1] = (
                variance[:, t] + p.kappa * (p.theta - vpos) * dt
                + p.xi * np.sqrt(vpos * dt) * z_v
            )
        return np.exp(log_s), variance


if __name__ == "__main__":
    model = HestonModel(HestonParameters())
    dist = model.terminal_distribution(n_qubits=8)
    print("Parameters:", model.params)
    print("Feller condition:", model.params.feller_condition)
    print("Price bins:", len(dist.probabilities))
    print("Probability sum:", dist.probabilities.sum())
    print("Amplitude norm:", np.linalg.norm(dist.amplitudes))
