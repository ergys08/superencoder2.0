"""Rank-by-rank oracle transport to an arbitrary Heston model in ONE shared SVD basis.

No neural network: project Q_target-Q_reference on the saved, globally ordered,
reference-anchored SVD basis. Compare fidelity for every prefix of that basis.

IMPORTANT: A basis fitted on one sampling region is not automatically accurate
outside it. 'Minimum required modes' always means minimum observed fidelity
threshold among the saved basis prefixes, not the intrinsic target dimension.
Run from repository root, e.g.:
  python -m transport_modes.rank_profile \
    --basis transport_modes/results/main/transport_basis.npz \
    --eta 0.05 0.04 2.0 0.04 0.30 -0.70 \
    --thresholds 0.99 0.999 0.9999
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .heston_quantiles import (
    PARAMETER_NAMES, QuantileConfig, heston_quantiles,
    quantiles_to_probabilities, validate_eta,
)
from .discover_transport_modes import fidelity, project
from .fit_transport import load_basis, out_of_range


def rank_profile(eta, basis, thresholds=(0.99, 0.999, 0.9999), max_modes=None):
    """Evaluate every prefix of a fixed shared basis against a fresh target law.

    Uses a SINGLE Heston FFT and no fitting or optimisation of the basis.
    The target-dependent coefficients are exact weighted projections (oracle).
    Unlike W2 projection error, rebinned amplitude fidelity need not grow
    monotonically with rank. Therefore every threshold is checked explicitly.
    """
    eta = validate_eta(eta)
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds or any(not np.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError("Thresholds must lie in (0,1].")
    modes = np.asarray(basis["modes"], dtype=float)
    limit = len(modes) if max_modes is None else int(max_modes)
    if limit < 0 or limit > len(modes):
        raise ValueError("max_modes must be between zero and number of stored modes")
    cfg = QuantileConfig(
        s0=float(basis["s0"]), maturity=float(basis["maturity"]),
        n_fft=int(basis["n_fft"]), n_quantiles=int(basis["n_quantiles"]),
        n_bins=int(basis["n_bins"]), x_half_width=float(basis["x_half_width"]),
        min_retained_mass=float(basis["min_retained_mass"]),
        max_negative_mass=float(basis["max_negative_mass"]),
    )
    target = heston_quantiles(eta, cfg)
    if not np.allclose(target.u, basis["u_grid"], atol=1e-14, rtol=0):
        raise ValueError("Quantile grids disagree")
    if not np.allclose(target.bin_edges, basis["bin_edges"], atol=1e-12, rtol=0):
        raise ValueError("Price-bin grids disagree")

    refq = np.asarray(basis["reference_quantiles"], dtype=float)
    weights = np.asarray(basis["weights"], dtype=float)
    delta = target.quantiles - refq
    # A single coefficient vector in the COMMON basis for every target.
    all_coeff = (delta * weights) @ modes[:limit].T
    total_energy = float(np.dot(weights, delta * delta))
    rows = []
    for k in range(limit + 1):
        # Equivalent to project(refq,target.quantiles,modes,weights,k).
        q_approx = refq + all_coeff[:k] @ modes[:k]
        probs, repaired_q, correction = quantiles_to_probabilities(
            q_approx, target.u, target.bin_edges
        )
        fid = fidelity(target.probabilities, probs)
        residual = float(np.dot(weights, (delta - (q_approx-refq))**2))
        corrected_w2 = float(np.sqrt(np.dot(
            weights, (target.quantiles - repaired_q)**2
        )))
        rows.append({
            "n_modes": k,
            "amplitude_fidelity": fid,
            "infidelity": 1.0-fid,
            "weighted_projection_residual": residual,
            "fraction_displacement_energy_captured": (
                1.0 - residual/total_energy if total_energy > 1e-25 else 1.0
            ),
            "weighted_quantile_w2_error_after_correction": corrected_w2,
            "isotonic_correction_required": bool(correction),
            "new_mode_coefficient": float(all_coeff[k-1]) if k else None,
        })

    requirements = {}
    for threshold in thresholds:
        passing = [r["n_modes"] for r in rows
                   if r["amplitude_fidelity"] >= threshold - 1e-14]
        requirements[str(threshold)] = {
            "attained": bool(passing),
            "minimum_modes": min(passing) if passing else None,
            "best_fidelity": max(r["amplitude_fidelity"] for r in rows),
        }
    bounds = out_of_range(eta, np.asarray(basis["bounds"], dtype=float))
    reference = np.asarray(basis["reference_eta"], dtype=float)
    dist = float(np.max(np.abs(eta-reference) /
                        (basis["bounds"][:, 1]-basis["bounds"][:, 0])))
    warning = None
    if not bounds["within_fitted_parameter_bounds"]:
        warning = ("EXTRAPOLATION: the target lies outside the region used to "
                   "discover the common SVD basis; fitted accuracy must be "
                   "measured and is not guaranteed.")
    return {
        "report": {
            "target": dict(zip(PARAMETER_NAMES, map(float, eta))),
            "reference": dict(zip(PARAMETER_NAMES, map(float, reference))),
            "basis_method": "reference-anchored, uncentred weighted quantile SVD",
            "stored_modes": len(modes),
            "evaluated_modes": limit,
            "thresholds": requirements,
            "normalised_linf_distance_from_reference": dist,
            **bounds,
            "warning": warning,
            "interpretation": (
                "Classical oracle projection: obtaining coefficients required "
                "the exact target Heston quantiles. No quantum unitary or "
                "parameter-to-coefficient predictor was constructed."
            ),
        },
        "ranks": rows,
        "coefficients": all_coeff,
        "target_quantiles": target.quantiles,
        "target_probabilities": target.probabilities,
    }


def save_profile(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rank_profile.json").write_text(
        json.dumps({"summary":result["report"], "ranks":result["ranks"]},indent=2)
    )
    with (out / "rank_profile.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["ranks"][0]))
        writer.writeheader()
        writer.writerows(result["ranks"])
    np.savez_compressed(out / "target_projection.npz",
                        coefficients=result["coefficients"],
                        target_quantiles=result["target_quantiles"],
                        target_probabilities=result["target_probabilities"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["n_modes"] for r in result["ranks"]]
    y = [r["amplitude_fidelity"] for r in result["ranks"]]
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    ax.plot(x, y, ".-")
    for threshold in result["report"]["thresholds"]:
        ax.axhline(float(threshold), linestyle="--",alpha=.45,
                   label=f"target F ≥ {threshold}")
    ax.set(xlabel="Number of common SVD modes",
           ylabel="Amplitude fidelity",
           title="Reference-to-target transport (oracle projection)")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fidelity_by_mode.png",dpi=160)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis",type=Path,required=True)
    parser.add_argument("--eta",nargs=6,type=float,required=True,
                        metavar=tuple(s.upper() for s in PARAMETER_NAMES))
    parser.add_argument("--thresholds",nargs="+",type=float,
                        default=[0.99,0.999,0.9999])
    parser.add_argument("--max-modes",type=int,default=None)
    parser.add_argument("--out",type=Path,
                        default=Path("transport_modes/results/rank_profile"))
    args=parser.parse_args()
    result=rank_profile(args.eta,load_basis(args.basis),
                        args.thresholds,args.max_modes)
    save_profile(result,args.out)
    print(json.dumps({"saved_to":str(args.out),**result["report"]},indent=2))


if __name__=="__main__":
    main()
