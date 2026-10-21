#!/usr/bin/env python3
"""Naive tangent-generator study for the Heston initial-variance family.

Run from the root of superencoder2.0:
    python "heston_coninuity/v0_naive_tangent_generators.py"

The same fixed Fourier grid, encoding cutoff, and 2**n log-price bins are
used for every v0. Only the *normalised, in-register* amplitudes are steered;
escaped tail mass is reported separately. The target tangents come from
finite differences and are cross-checked using the exact derivative of the
Heston characteristic function with respect to v0.

For each v0, project the state tangent onto directions -i P |psi> for Pauli
strings P with odd Y parity and contiguous support span <= r, r=1..4.
We report (a) the unrestricted least-squares residual, (b) the residual
from a fixed-size greedy subset reselected at each target, and (c) the residual
from a subset selected once at the v0 reference and frozen across the sweep.
These are *local* tangent fits; they are NOT finite-displacement transport
circuits or global reachability certificates. No claim of exponential decay
or MPS assumptions is made.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import lstsq as scipy_lstsq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heston import HestonModel, HestonParameters

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


@dataclass(frozen=True)
class Settings:
    qubits: int = 8
    n_fft: int = 16384
    half_width: float = 2.5
    cutoff: float = 0.7
    v0_min: float = 0.01
    v0_max: float = 0.20
    v0_points: int = 21
    reference_v0: float = 0.04
    epsilon: float = 1e-4
    max_support: int = 4
    budget: int = 16
    output: str = "heston_coninuity/results/v0_naive_tangent_generators"


BASE = HestonParameters(s0=100.0, mu=0.05, v0=0.04, kappa=2.0,
                        theta=0.04, xi=0.30, rho=-0.70, maturity=1.0)


class HestonFamily:
    def __init__(self, settings: Settings):
        self.s = settings
        self.edges = np.linspace(np.log(BASE.s0) - settings.cutoff,
                                 np.log(BASE.s0) + settings.cutoff,
                                 (1 << settings.qubits) + 1)
        self.dx = 2 * settings.half_width / settings.n_fft
        self.origin = np.log(BASE.s0) - settings.half_width
        self.freq = 2 * np.pi * np.fft.fftfreq(settings.n_fft, d=self.dx)
        self.fft_edges = self.origin + self.dx * np.arange(settings.n_fft + 1)
        self.phase = np.exp(-1j * self.freq * self.origin)
        self.cache = {}

    def probabilities_and_derivative(self, v0: float):
        key = round(float(v0), 12)
        if key in self.cache:
            return self.cache[key]
        params = replace(BASE, v0=float(v0))
        model = HestonModel(params)
        u = self.freq
        iu = 1j * u
        b = params.kappa - params.rho * params.xi * iu
        d = np.sqrt(b*b + params.xi**2 * (u*u + iu))
        g = (b-d) / (b+d)
        z = np.exp(-d * params.maturity)
        D = ((b-d) / params.xi**2) * (1-z) / (1-g*z)
        phi = model.characteristic_function(u)
        raw = np.real(np.fft.fft(phi * self.phase)) / (self.s.n_fft * self.dx)
        draw = np.real(np.fft.fft(D * phi * self.phase)) / (self.s.n_fft * self.dx)
        negative_mass = float(np.maximum(-raw, 0).sum() * self.dx)
        if negative_mass > 1e-4:
            raise ArithmeticError(f"Large FFT negative mass: {negative_mass:.3g}")
        mask = raw > 0
        masses = np.where(mask, raw, 0.0) * self.dx
        dmasses = np.where(mask, draw, 0.0) * self.dx
        total, dtotal = float(masses.sum()), float(dmasses.sum())
        if total <= 0:
            raise ArithmeticError("Fourier density has zero mass")
        pmass = masses / total
        dpmass = (dmasses * total - masses * dtotal) / total**2
        edge_fraction = float(pmass[:self.s.n_fft//100].sum() +
                              pmass[-self.s.n_fft//100:].sum())
        if edge_fraction > 2e-3:
            raise ArithmeticError(f"Fourier inversion window too narrow: {edge_fraction:.3g}")
        cdf = np.r_[0.0, np.cumsum(pmass)]
        dcdf = np.r_[0.0, np.cumsum(dpmass)]
        raw_bins = np.maximum(np.diff(np.interp(self.edges, self.fft_edges, cdf)), 0)
        dr_bins = np.diff(np.interp(self.edges, self.fft_edges, dcdf))
        kept = float(raw_bins.sum())
        dkept = float(dr_bins.sum())
        if kept <= 0 or np.any(raw_bins <= 0):
            raise ArithmeticError("Found empty/zero bin; increase cutoff reliability or FFT accuracy")
        p = raw_bins / kept
        dp = (dr_bins * kept - raw_bins * dkept) / kept**2
        amplitude = np.sqrt(p)
        tangent = dp / (2 * amplitude)
        tangent -= amplitude * np.dot(amplitude, tangent)
        result = (amplitude, tangent, kept, 1-kept, edge_fraction, negative_mass)
        self.cache[key] = result
        return result

    def psi(self, v0: float):
        return self.probabilities_and_derivative(v0)[0]


def pauli_words(n: int, max_span: int):
    """Odd-Y Pauli words; support is smallest containing contiguous interval."""
    grouped = {r: [] for r in range(1, max_span + 1)}
    for span in grouped:
        for start in range(n-span+1):
            for block in product("IXYZ", repeat=span):
                if block[0] == "I" or block[-1] == "I":
                    continue
                if sum(x == "Y" for x in block) % 2 != 1:
                    continue
                word = "I"*start + "".join(block) + "I"*(n-start-span)
                grouped[span].append(word)
    return grouped


def actions(psi: np.ndarray, words: list[str]):
    """Return columns -i P |psi> in big-endian qubit ordering."""
    n = int(np.log2(len(psi)))
    indices = np.arange(len(psi), dtype=np.int64)
    columns = np.empty((len(psi), len(words)), dtype=np.float64)
    for k, word in enumerate(words):
        xmask = 0
        zmask = 0
        ys = 0
        for pos, letter in enumerate(word):
            bit = 1 << (n-1-pos)
            if letter in "XY":
                xmask |= bit
            if letter in "YZ":
                zmask |= bit
            if letter == "Y":
                ys += 1
        if ys % 2 != 1:
            raise ValueError(f"Expected odd Y parity: {word}")
        source = indices ^ xmask
        parity = np.zeros_like(indices)
        for bit in range(n):
            if zmask & (1 << bit):
                parity ^= ((source >> bit) & 1)
        sign = (1 - 2*parity) * ((-1) ** ((ys-1)//2))
        columns[:, k] = sign * psi[source]
    return columns


def leakage(t: np.ndarray, columns: np.ndarray):
    """Relative squared residual after unconstrained least squares."""
    if columns.shape[1] == 0:
        return 1.0, 0
    coeff, _, rank, _ = scipy_lstsq(columns, t, cond=1e-10,
                                   lapack_driver="gelsy", check_finite=False)
    err = t - columns @ coeff
    return float(np.clip(np.dot(err, err) / np.dot(t, t), 0, 1)), int(rank)


def greedy_leakage(t: np.ndarray, columns: np.ndarray, budget: int):
    """Orthogonal matching pursuit: a fixed-size selected-generator baseline."""
    norms = np.linalg.norm(columns, axis=0)
    eligible = norms > 1e-12
    X = np.zeros_like(columns)
    X[:, eligible] = columns[:, eligible] / norms[eligible]
    selected = []
    residual = t.copy()
    for _ in range(min(budget, columns.shape[1])):
        correlations = np.abs(X.T @ residual)
        correlations[~eligible] = -1.0
        if selected:
            correlations[selected] = -1.0
        idx = int(np.argmax(correlations))
        if correlations[idx] < 1e-10 * np.linalg.norm(t):
            break
        selected.append(idx)
        coef, _, _, _ = scipy_lstsq(X[:, selected], t, cond=1e-10,
                                    lapack_driver="gelsy", check_finite=False)
        residual = t - X[:, selected] @ coef
    value = float(np.clip(np.dot(residual, residual) / np.dot(t, t), 0, 1))
    return value, selected


def draw_plots(rows: list[dict], settings: Settings, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v0 = sorted(set(r["v0"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), constrained_layout=True)
    for support in range(1, settings.max_support+1):
        data = [next(row for row in rows if row["v0"] == v and row["support"] == support)
                for v in v0]
        axes[0].plot(v0, [x["unrestricted_leakage"] for x in data],
                     marker=".", label=f"support ≤ {support} ({data[0]['library_size']} strings)")
        axes[1].plot(v0, [x["budget_leakage"] for x in data],
                     marker=".", label=f"support ≤ {support}")
    for ax in axes:
        ax.set_xlabel("Heston initial variance v₀")
        ax.set_ylabel("Relative squared tangent leakage")
        ax.set_ylim(-.025, 1.025)
        ax.grid(alpha=.22)
        ax.axvline(settings.reference_v0, linestyle=":", linewidth=1.5,
                   label="reference v₀" if ax is axes[0] else None)
        ax.legend(fontsize=8)
    axes[0].set_title("All eligible Pauli directions (no gate budget)")
    axes[1].set_title(f"Greedy subset: at most {settings.budget} Pauli directions")
    for ax in axes:
        ax.set_yscale("symlog", linthresh=1e-7)
        ax.set_ylim(0, 1.1)
    fig.suptitle("Heston v₀ quantum-state tangent approximation (8-qubit terminal law)")
    fig.savefig(out/"leakage_vs_v0.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    for support in range(1, settings.max_support+1):
        data = [next(row for row in rows if row["v0"] == v and row["support"] == support)
                for v in v0]
        ax.plot(v0, [x["reference_fixed_leakage"] for x in data], marker=".",
                label=f"support ≤ {support}")
    ax.axvline(settings.reference_v0, color="grey", linestyle=":", label="reference v₀")
    ax.set_xlabel("Heston initial variance v₀")
    ax.set_ylabel("Relative squared tangent leakage")
    ax.set_yscale("log")
    ax.grid(alpha=.2, which="both")
    ax.legend()
    ax.set_title(f"Same ≤{settings.budget} Pauli strings at every v₀ (selected at reference)")
    fig.savefig(out/"fixed_reference_generators.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    base = [next(row for row in rows if row["v0"] == v and row["support"] == 1)
            for v in v0]
    ax.plot(v0, [r["tangent_norm"] for r in base], label="||∂ψ/∂v₀||")
    ax.set_xlabel("Heston initial variance v₀")
    ax.set_ylabel("Tangent norm")
    ax.grid(alpha=.25)
    a2 = ax.twinx()
    a2.plot(v0, [r["lost_mass"] for r in base], linestyle="--", label="Mass outside cutoff")
    a2.set_ylabel("Discarded probability mass")
    lines = ax.get_lines() + a2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    ax.set_title("State sensitivity and finite-register cutoff loss")
    fig.savefig(out/"tangent_norm_and_cutoff.png", dpi=170)
    plt.close(fig)


def run(settings: Settings):
    if not 2 <= settings.qubits <= 12 or not 1 <= settings.max_support <= settings.qubits:
        raise ValueError("Invalid qubits or max support")
    if settings.n_fft < (1 << settings.qubits) or (settings.n_fft & (settings.n_fft-1)):
        raise ValueError("n_fft must be a power of two and >= number of bins")
    if settings.cutoff <= 0 or settings.cutoff >= settings.half_width * 0.9:
        raise ValueError("The cutoff must be inside a substantially wider FFT window")
    if settings.v0_min <= 2*settings.epsilon or settings.v0_max <= settings.v0_min:
        raise ValueError("v0 range/epsilon invalid")
    if settings.budget < 1 or settings.v0_points < 2:
        raise ValueError("Invalid budget or number of sweep points")

    out = Path(settings.output)
    out.mkdir(parents=True, exist_ok=True)
    family = HestonFamily(settings)
    grouped = pauli_words(settings.qubits, settings.max_support)
    words = []
    libraries = {}
    for r in range(1, settings.max_support+1):
        words += grouped[r]
        libraries[r] = list(words)

    # Discover one immutable generator subset at the reference.
    ref_psi, ref_tangent, *_ = family.probabilities_and_derivative(settings.reference_v0)
    reference_selected = {}
    for support, library in libraries.items():
        _, indices = greedy_leakage(ref_tangent, actions(ref_psi, library), settings.budget)
        reference_selected[support] = indices
    (out/"fixed_reference_words.json").write_text(json.dumps({
        str(r): [libraries[r][k] for k in indices]
        for r, indices in reference_selected.items()
    }, indent=2) + "\n")

    v0_values = sorted(set(np.round(np.r_[np.linspace(settings.v0_min, settings.v0_max,
                                                       settings.v0_points),
                                              settings.reference_v0], 8).tolist()))
    rows = []
    for j, v0 in enumerate(v0_values, start=1):
        psi, tangent, kept, lost, edge, neg = family.probabilities_and_derivative(v0)
        eps = settings.epsilon
        tangent_fd = (family.psi(v0+eps)-family.psi(v0-eps))/(2*eps)
        tangent_fd_half = (family.psi(v0+eps/2)-family.psi(v0-eps/2))/eps
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-10:
            raise ArithmeticError(f"Tangent nearly zero at v0={v0}")
        fd_error = float(np.linalg.norm(tangent-tangent_fd)/norm)
        fd_half_error = float(np.linalg.norm(tangent-tangent_fd_half)/norm)
        fd_self_error = float(np.linalg.norm(tangent_fd_half-tangent_fd)/norm)
        A = actions(psi, libraries[settings.max_support])
        for support in libraries:
            size = len(libraries[support])
            span_actions = A[:, :size]
            unrestricted, rank = leakage(tangent, span_actions)
            budget_value, selected = greedy_leakage(tangent, span_actions, settings.budget)
            ref_fixed, _ = leakage(tangent, span_actions[:, reference_selected[support]])
            rows.append({
                "v0": v0,
                "support": support,
                "library_size": size,
                "tangent_space_rank": rank,
                "unrestricted_leakage": unrestricted,
                "budget_leakage": budget_value,
                "reference_fixed_leakage": ref_fixed,
                "generators_selected": len(selected),
                "selected_words": " ".join(libraries[support][i] for i in selected),
                "tangent_norm": norm,
                "kept_mass": kept,
                "lost_mass": lost,
                "fft_edge_mass": edge,
                "fft_negative_mass_clipped": neg,
                "fd_relative_error_epsilon": fd_error,
                "fd_relative_error_epsilon_over_2": fd_half_error,
                "fd_step_difference_relative": fd_self_error,
            })
        print(f"[{j:2d}/{len(v0_values)}] v0={v0:.4f}, "
              f"leakage(s1,s2,s3,s4)=" + ",".join(
                  f"{rows[-settings.max_support+k]['budget_leakage']:.4f}"
                  for k in range(settings.max_support)) +
              f", tangent FD discrepancy={fd_half_error:.2e}", flush=True)

    with (out/"leakage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out/"settings.json").write_text(json.dumps(vars(settings), indent=2) + "\n")
    draw_plots(rows, settings, out)
    summary = {}
    for support in libraries:
        subset = [r for r in rows if r["support"] == support]
        summary[str(support)] = {
            "library_size": len(libraries[support]),
            "rank_at_reference": next(r["tangent_space_rank"] for r in subset
                                      if r["v0"] == settings.reference_v0),
            "mean_unrestricted_leakage": float(np.mean([r["unrestricted_leakage"] for r in subset])),
            "mean_budget_leakage": float(np.mean([r["budget_leakage"] for r in subset])),
            "max_budget_leakage": float(max(r["budget_leakage"] for r in subset)),
            "mean_reference_fixed_leakage": float(np.mean([r["reference_fixed_leakage"] for r in subset])),
            "max_reference_fixed_leakage": float(max(r["reference_fixed_leakage"] for r in subset)),
        }
    summary["derivative_max_relative_error"] = max(r["fd_relative_error_epsilon_over_2"] for r in rows)
    (out/"summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nSupport summary:", json.dumps(summary, indent=2))
    print(f"\nFiles written to: {out.resolve()}")
    return rows


def main():
    s = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=s.output)
    parser.add_argument("--qubits", type=int, default=s.qubits)
    parser.add_argument("--n-fft", type=int, default=s.n_fft)
    parser.add_argument("--cutoff", type=float, default=s.cutoff)
    parser.add_argument("--v0-min", type=float, default=s.v0_min)
    parser.add_argument("--v0-max", type=float, default=s.v0_max)
    parser.add_argument("--v0-points", type=int, default=s.v0_points)
    parser.add_argument("--epsilon", type=float, default=s.epsilon)
    parser.add_argument("--max-support", type=int, default=s.max_support)
    parser.add_argument("--budget", type=int, default=s.budget)
    args = parser.parse_args()
    s = replace(s, output=args.output, qubits=args.qubits, n_fft=args.n_fft,
                cutoff=args.cutoff, v0_min=args.v0_min, v0_max=args.v0_max,
                v0_points=args.v0_points, epsilon=args.epsilon,
                max_support=args.max_support, budget=args.budget)
    if threadpool_limits:
        with threadpool_limits(limits=1, user_api="blas"):
            run(s)
    else:
        run(s)


if __name__ == "__main__":
    main()
