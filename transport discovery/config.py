"""Reproducible domain and experiment settings.

The bounds are copied from ergys08/optimal-transport, 13 October 2026.
They are not a calibration to market prices.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PARAMETER_NAMES = ('mu', 'v0', 'kappa', 'theta', 'xi', 'rho')
BOUNDS = np.array([
    [-0.03, 0.12],
    [0.01, 0.10],
    [0.80, 4.00],
    [0.02, 0.10],
    [0.15, 0.75],
    [-0.90, -0.10],
], dtype=np.float64)
REFERENCE = BOUNDS.mean(axis=1)


@dataclass(frozen=True)
class Settings:
    """All random choices and evaluation budgets are controlled here."""
    n_qubits: int = 8
    n_fft: int = 4096
    seed: int = 2026
    n_train: int = 192
    n_validation: int = 64
    n_test: int = 64
    n_candidates: int = 60
    n_keep_oracle: int = 12
    n_finalists: int = 3
    max_primitives: int = 20
    max_cx: int = 60
    max_support: int = 4
    oracle_states: int = 12
    oracle_restarts: int = 2
    oracle_maxiter: int = 80
    nn_epochs: int = 130
    nn_width: int = 48
    nn_batch: int = 32
    nn_patience: int = 22
    target_fidelity: float = 0.9999
    # Fraction of highest-infidelity examples included in each CVaR-like loss.
    nn_tail_fraction: float = 0.25
    nn_tail_weight: float = 0.75
    # Finalist SELECTION probes: all corners + seeded interior + adversarial.
    robustness_validation_interior: int = 128
    robustness_validation_adversarial_iters: int = 6
    robustness_validation_adversarial_popsize: int = 4
    # Final WINNER audit: different random/adversarial seed, no re-selection.
    robustness_test_interior: int = 512
    robustness_test_adversarial_iters: int = 18
    robustness_test_adversarial_popsize: int = 6

    def validate(self) -> None:
        if not 2 <= self.n_qubits <= 12:
            raise ValueError('Use 2 to 12 qubits in this classical experiment')
        if self.n_fft < 2**self.n_qubits or self.n_fft & (self.n_fft-1):
            raise ValueError('n_fft must be a power of two >= 2**n_qubits')
        if min(self.n_train, self.n_validation, self.n_test) < 2:
            raise ValueError('Each state-level split needs >= 2 states')
        if min(self.n_candidates, self.n_keep_oracle, self.n_finalists) < 1:
            raise ValueError('Search budgets must be positive')
        if not (1 <= self.max_support <= self.n_qubits):
            raise ValueError('max_support must be between 1 and n_qubits')
        if not (0 < self.target_fidelity <= 1):
            raise ValueError('Invalid fidelity target')
        if not (0 < self.nn_tail_fraction <= 1 and 0 <= self.nn_tail_weight <= 1):
            raise ValueError('Invalid NN tail loss configuration')
        budgets = (self.robustness_validation_interior, self.robustness_test_interior,
                   self.robustness_validation_adversarial_iters,
                   self.robustness_test_adversarial_iters,
                   self.robustness_validation_adversarial_popsize,
                   self.robustness_test_adversarial_popsize)
        if min(budgets) < 0:
            raise ValueError('Robustness budgets must be nonnegative')