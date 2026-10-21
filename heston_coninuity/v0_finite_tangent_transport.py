#!/usr/bin/env python3
"""Finite, statevector-based Heston v0 transport using tangent-fit Pauli generators.

Run from the superencoder2.0 root:
    python "heston_coninuity/v0_finite_tangent_transport.py"

The exact (up to FFT discretisation) target curve is the normalised eight-qubit
terminal-price amplitude encoding, with fixed log-price bins and explicit tail
loss. The exact tangent is obtained by Fourier inversion of the analytical
Heston characteristic-function derivative (not by sampling).

Choose ONE fixed set of low-support Hermitian Pauli strings. At each midpoint
in v0, least-squares-fit their coefficients to the *exact target tangent* and
apply the corresponding physical unitary to the PROPAGATED state, not to the
exact target. This tests finite-path accumulation of steering error.

Two fixed-set selection strategies:
    reference: greedy selection using only the v0=0.04 tangent;
    trajectory: joint greedy selection over a finite training grid (an oracle
                architecture screening baseline, not a deployable algorithm).
The strings stay fixed; their coefficients vary along the trajectory.

The target derivative is used at EVERY integration step. Thus this is an
oracle-driven finite transport experiment, not yet an NN that predicts
coefficients from parameters. Scores measure ideal noiseless unitaries. Gate
cost/compilation and finite-data training are not included. A failure at
finite optimisation precision is not a proven no-go theorem.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from math import ceil
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v0_naive_tangent_generators import (HestonFamily, Settings as FamilySettings,
    actions, pauli_words, greedy_leakage)

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


@dataclass(frozen=True)
class Settings:
    reference_v0: float = 0.04
    v0_min: float = 0.01
    v0_max: float = 0.20
    step: float = 0.002
    qubits: int = 8
    n_fft: int = 16384
    cutoff: float = 0.7
    max_support: int = 3
    budgets: tuple[int, ...] = (4, 8, 16)
    methods: tuple[str, ...] = ("reference", "trajectory")
    architecture_train_points: int = 13
    fidelity_threshold: float = 0.999
    leakage_threshold: float = 0.05
    output: str = "heston_coninuity/results/v0_finite_tangent_transport"


def matrix_of_generator(word: str) -> csr_matrix:
    """Real skew-symmetric K=-iP for a Pauli word with odd Y count."""
    n = len(word)
    dimension = 1 << n
    rows = np.arange(dimension, dtype=np.int64)
    x_mask = 0
    z_mask = 0
    y_count = 0
    for pos, letter in enumerate(word):
        bit = 1 << (n-1-pos)
        if letter in "XY":
            x_mask |= bit
        if letter in "YZ":
            z_mask |= bit
        if letter == "Y":
            y_count += 1
    if y_count % 2 != 1:
        raise ValueError(f"Expected odd number of Ys in {word}")
    cols = rows ^ x_mask
    parity = np.zeros(dimension, dtype=np.int64)
    for bit in range(n):
        if z_mask & (1 << bit):
            parity ^= (cols >> bit) & 1
    signs = ((1-2*parity) * (-1)**((y_count-1)//2)).astype(float)
    K = csr_matrix((signs, (rows, cols)), shape=(dimension, dimension))
    if (K+K.T).nnz != 0:
        raise AssertionError(f"Generator is not antisymmetric: {word}")
    return K


def joint_greedy(psi_samples: list[np.ndarray], tangent_samples: list[np.ndarray],
                 words: list[str], budget: int) -> list[int]:
    """Choose one pool to minimise mean RELATIVE tangent leakage on train grid.

    At every training point coefficients are re-fitted independently. No NN is
    trained, and the full target derivatives are used for library selection.
    """
    A = np.stack([actions(psi, words) for psi in psi_samples], axis=0)
    T = np.stack([t/np.linalg.norm(t) for t in tangent_samples], axis=0)
    norms = np.linalg.norm(A, axis=1)
    A = A / np.maximum(norms[:, None, :], 1e-13)
    residual = T.copy()
    picked = []
    for _ in range(min(budget, len(words))):
        correlations = np.einsum("sdi,sd->si", A, residual, optimize=True)
        score = np.einsum("si,si->i", correlations, correlations, optimize=True)
        if picked:
            score[picked] = -1.0
        choice = int(np.argmax(score))
        if score[choice] < 1e-18:
            break
        picked.append(choice)
        for sample in range(len(T)):
            coeff = np.linalg.lstsq(A[sample][:, picked], T[sample], rcond=1e-10)[0]
            residual[sample] = T[sample] - A[sample][:, picked] @ coeff
    return picked


def make_paths(settings: Settings):
    paths = {}
    for branch, end in (("decreasing", settings.v0_min),
                        ("increasing", settings.v0_max)):
        intervals = int(ceil(abs(end-settings.reference_v0)/settings.step))
        paths[branch] = np.linspace(settings.reference_v0, end, intervals+1)
    return paths


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_failure(rows: list[dict], metric: str, threshold: float):
    for row in rows[1:]:
        if (row[metric] < threshold if metric == "fidelity" else row[metric] > threshold):
            return float(row["v0"])
    return None


def create_figures(trajectory_rows: list[dict], selections: dict,
                   coefficient_rows: list[dict], settings: Settings, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5), constrained_layout=True)
    # A legible subset: 2- and 3-qubit local spans, comparing fixed budgets,
    # plus the reference-selected 16-direction baseline.
    cases = [(s, b, "trajectory") for s in (2, 3) for b in settings.budgets]
    cases += [(s, 16, "reference") for s in (2,3) if 16 in settings.budgets]
    for support, budget, mode in cases:
        data = [r for r in trajectory_rows if
                (r["support"],r["budget"],r["method"]) == (support,budget,mode)]
        if not data:
            continue
        # Both paths include the reference, so remove one duplicate.
        data = {float(r["v0"]):r for r in data}
        subset = [data[v] for v in sorted(data)]
        label = f"≤{support} sites, K={budget} ({mode})"
        style = "--" if mode == "reference" else "-"
        axes[0].plot([r["v0"] for r in subset],
                     [max(1-r["fidelity"],1e-12) for r in subset],
                     linestyle=style, linewidth=1.7, label=label)
        axes[1].plot([r["v0"] for r in subset],
                     [max(r["local_leakage"],1e-12) for r in subset],
                     linestyle=style, linewidth=1.7, label=label)
    axes[0].axhline(1-settings.fidelity_threshold, color="red", linestyle=":",
                    label="endpoint fidelity threshold")
    axes[1].axhline(settings.leakage_threshold, color="red", linestyle=":",
                    label="local leakage threshold")
    for ax in axes:
        ax.axvline(settings.reference_v0, color="grey", linestyle=":")
        ax.set_xlabel("Heston initial variance v₀")
        ax.set_yscale("log")
        ax.grid(alpha=.24, which="both")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Accumulated endpoint infidelity 1−F")
    axes[1].set_ylabel("Relative squared tangent leakage at path step")
    fig.suptitle("Finite unitary transport: a fixed Pauli pool, refitted coefficients")
    fig.savefig(output/"finite_transport_vs_v0.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5), constrained_layout=True)
    for support in (2, 3):
        for budget in settings.budgets:
            data = [r for r in trajectory_rows if
                    (r["support"],r["budget"],r["method"]) ==
                    (support,budget,"trajectory")]
            data = {float(r["v0"]):r for r in data}
            subset = [data[v] for v in sorted(data)]
            if subset:
                ax.plot([r["v0"] for r in subset], [r["fidelity"] for r in subset],
                        label=f"support≤{support}, K={budget}")
    ax.axhline(settings.fidelity_threshold, color="red", linestyle=":",
               label=f"threshold {settings.fidelity_threshold}")
    ax.axvline(settings.reference_v0, color="grey", linestyle=":")
    ax.set(xlabel="Heston initial variance v₀", ylabel="Fidelity of propagated state",
           title="Transportable interval by generator count and support")
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.savefig(output/"fidelity_by_budget.png", dpi=180)
    plt.close(fig)

    # Show actual Hermitian Pauli strings and coefficient functions for one
    # interpretable, small fixed pool (two-site support, 8 generators).
    key=(2, 8, "trajectory")
    if key in selections:
        selected=selections[key]
        fig, ax = plt.subplots(figsize=(10.7, 5.1), constrained_layout=True)
        for word in selected:
            data = sorted((r for r in coefficient_rows if
                (r["support"],r["budget"],r["method"],r["branch"],r["pauli_string"]) ==
                (2,8,"trajectory","increasing",word)),
                key=lambda r:r["v0_midpoint"])
            ax.plot([r["v0_midpoint"] for r in data],
                    [r["coefficient"] for r in data], label=word)
        ax.set(xlabel="Heston initial variance v₀ (step midpoint)",
               ylabel="Generator coefficient c(v₀)",
               title="Actual 8-string, two-site generator: H(v₀)=Σ cₖ(v₀)Pₖ")
        ax.grid(alpha=.25);ax.legend(ncol=2, fontsize=9)
        fig.savefig(output/"eight_generator_coefficients.png",dpi=180)
        plt.close(fig)


def run(settings: Settings):
    if not (0 < settings.v0_min < settings.reference_v0 < settings.v0_max):
        raise ValueError("Require positive v0_min < reference_v0 < v0_max")
    if settings.step <= 0 or settings.architecture_train_points < 2:
        raise ValueError("Invalid integration step or architecture training size")
    if settings.max_support > settings.qubits or settings.max_support < 1:
        raise ValueError("Invalid local generator support")
    if not 0 < settings.fidelity_threshold < 1 or not 0 < settings.leakage_threshold < 1:
        raise ValueError("Thresholds must be strictly inside (0,1)")
    out = Path(settings.output)
    out.mkdir(parents=True, exist_ok=True)
    family = HestonFamily(FamilySettings(qubits=settings.qubits, n_fft=settings.n_fft,
                                         cutoff=settings.cutoff, max_support=settings.max_support,
                                         reference_v0=settings.reference_v0))
    groups = pauli_words(settings.qubits, settings.max_support)
    pools = {}
    words = []
    for span in range(1, settings.max_support+1):
        words += groups[span]
        pools[span] = list(words)
    reference_psi, reference_t, *_ = family.probabilities_and_derivative(settings.reference_v0)
    training_grid = np.linspace(settings.v0_min, settings.v0_max, settings.architecture_train_points)
    training_psi, training_t = [], []
    for x in training_grid:
        p,t,*_ = family.probabilities_and_derivative(float(x))
        training_psi.append(p); training_t.append(t)
    selections = {}
    matrices = {}
    for support, library in pools.items():
        ref_matrix = actions(reference_psi, library)
        for budget in sorted(set(settings.budgets)):
            for method in settings.methods:
                if method == "reference":
                    _, indices = greedy_leakage(reference_t, ref_matrix, budget)
                elif method == "trajectory":
                    indices = joint_greedy(training_psi, training_t, library, budget)
                else:
                    raise ValueError("Unknown architecture selection method: "+method)
                selections[(support, budget, method)] = [library[i] for i in indices]
                print(f"Selected support≤{support} budget={budget} {method}: "
                      + ", ".join(library[i] for i in indices), flush=True)
                for word in selections[(support,budget,method)]:
                    if word not in matrices:
                        matrices[word] = matrix_of_generator(word)
    selection_rows = [
        {"support":s,"budget":b,"method":m,"generator_rank":i+1,
         "pauli_string":p,"nonidentity_factors":len(p)-p.count("I"),
         "support_span":(max(j for j,c in enumerate(p) if c != 'I')-min(j for j,c in enumerate(p) if c != 'I')+1)}
        for (s,b,m), lst in selections.items() for i,p in enumerate(lst)
    ]
    write_csv(out/"selected_generators.csv", selection_rows)
    paths = make_paths(settings)
    trajectory_rows, coefficient_rows = [], []
    for (support,budget,method), chosen_words in selections.items():
        K_matrices = [matrices[word] for word in chosen_words]
        for branch, path in paths.items():
            current = reference_psi.copy()
            for k, point in enumerate(path):
                point = float(point)
                exact, tangent, kept, lost, _, _ = family.probabilities_and_derivative(point)
                F = float(np.clip(np.dot(current, exact)**2, 0.0, 1.0))
                identity_fid = float(np.dot(reference_psi, exact)**2)
                if k == 0:
                    local_L = 0.0
                    next_l2 = 0.0
                else:
                    # This leakage measures the actual midpoint tangent fit used
                    # for the immediately preceding step (not the endpoint's fit).
                    local_L = last_local_L
                    next_l2 = float(np.linalg.norm(current-exact))
                trajectory_rows.append({"support":support,"budget":budget,"method":method,
                                        "branch":branch,"step_index":k,"v0":point,
                                        "fidelity":F,"infidelity":1-F,
                                        "identity_fidelity":identity_fid,
                                        "local_leakage":local_L,
                                        "state_vector_distance":next_l2,
                                        "kept_mass":kept,"lost_mass":lost,
                                        "number_generators":len(chosen_words)})
                if k == len(path)-1:
                    break
                nxt = float(path[k+1]); delta = nxt-point
                mid = .5 * (point+nxt)
                mid_psi, mid_t, *_ = family.probabilities_and_derivative(mid)
                tangent_columns = actions(mid_psi, chosen_words)
                coeff = np.linalg.lstsq(tangent_columns, mid_t, rcond=1e-10)[0]
                error = mid_t - tangent_columns @ coeff
                last_local_L = float(np.dot(error,error)/np.dot(mid_t,mid_t))
                for word, value in zip(chosen_words, coeff):
                    coefficient_rows.append({"support":support,"budget":budget,
                                             "method":method,"branch":branch,
                                             "v0_midpoint":mid,"pauli_string":word,
                                             "coefficient":float(value),
                                             "angle_for_this_step":float(2*delta*value),
                                             "local_leakage":last_local_L})
                G = sum((float(c)*K for c,K in zip(coeff, K_matrices)),
                        csr_matrix((len(current),len(current))))
                current = expm_multiply(delta * G, current, traceA=0.0)
                # exp of antisymmetric real K is an orthogonal matrix
                if abs(np.linalg.norm(current)-1)>1e-9:
                    raise AssertionError("Unitary integration did not preserve norm")
        print(f"Finished r≤{support}, K={budget}, {method}", flush=True)

    summary_rows=[]
    for (support,budget,method), selected in selections.items():
        for branch in paths:
            relevant=sorted((row for row in trajectory_rows if
                row["support"]==support and row["budget"]==budget and
                row["method"]==method and row["branch"]==branch),
                key=lambda row:row["step_index"])
            bad_F=first_failure(relevant,"fidelity",settings.fidelity_threshold)
            bad_L=first_failure(relevant,"local_leakage",settings.leakage_threshold)
            summary_rows.append({"support":support,"budget":budget,"method":method,
                "branch":branch,"generators":len(selected),
                "endpoint_v0":float(relevant[-1]["v0"]),
                "endpoint_fidelity":float(relevant[-1]["fidelity"]),
                "worst_fidelity":min(row["fidelity"] for row in relevant),
                "first_fidelity_failure_v0":bad_F,
                "first_local_leakage_failure_v0":bad_L,
                "max_local_leakage":max(row["local_leakage"] for row in relevant)})
    write_csv(out/"trajectory.csv",trajectory_rows)
    write_csv(out/"generator_coefficients.csv",coefficient_rows)
    write_csv(out/"boundary_summary.csv",summary_rows)
    (out/"settings.json").write_text(json.dumps(asdict(settings),indent=2)+"\n")
    (out/"generator_sets.json").write_text(json.dumps({
        f"support_{s}_budget_{b}_{m}":v
        for (s,b,m),v in selections.items()},indent=2)+"\n")
    create_figures(trajectory_rows,selections,coefficient_rows,settings,out)
    print("\nFinite-trajectory boundary summary:")
    for row in summary_rows:
        if row["method"]=="trajectory" and row["support"] in (2,3):
            print(f"r≤{row['support']} K={row['budget']:2d} {row['branch']:10s} "
                  f"endpoint F={row['endpoint_fidelity']:.7f} "
                  f"first F<{settings.fidelity_threshold} at {row['first_fidelity_failure_v0']} "
                  f"first L>{settings.leakage_threshold} at {row['first_local_leakage_failure_v0']}")
    print("Outputs:",out.resolve())
    return trajectory_rows,summary_rows


def main():
    defaults=Settings()
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=defaults.output)
    parser.add_argument("--step", type=float, default=defaults.step)
    parser.add_argument("--cutoff", type=float, default=defaults.cutoff)
    parser.add_argument("--qubits", type=int, default=defaults.qubits)
    parser.add_argument("--v0-min", type=float, default=defaults.v0_min)
    parser.add_argument("--v0-max", type=float, default=defaults.v0_max)
    parser.add_argument("--reference-v0", type=float, default=defaults.reference_v0)
    parser.add_argument("--max-support", type=int, default=defaults.max_support)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(defaults.budgets))
    parser.add_argument("--fidelity-threshold", type=float, default=defaults.fidelity_threshold)
    parser.add_argument("--leakage-threshold", type=float, default=defaults.leakage_threshold)
    parser.add_argument("--architecture-train-points",type=int,default=defaults.architecture_train_points)
    args=parser.parse_args()
    settings=replace(defaults, output=args.output,step=args.step,cutoff=args.cutoff,
        qubits=args.qubits, v0_min=args.v0_min, v0_max=args.v0_max,
        reference_v0=args.reference_v0, max_support=args.max_support,
        budgets=tuple(sorted(set(args.budgets))),
        fidelity_threshold=args.fidelity_threshold,
        leakage_threshold=args.leakage_threshold,
        architecture_train_points=args.architecture_train_points)
    if threadpool_limits is not None:
        with threadpool_limits(limits=1, user_api="blas"):
            run(settings)
    else:
        run(settings)


if __name__=="__main__":
    main()