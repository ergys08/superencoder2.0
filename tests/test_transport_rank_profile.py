"""Shared fixed SVD mode basis target-profile regression tests."""
import numpy as np
from five_mode_transport.discover_transport_modes import fit_reference_modes
from five_mode_transport.heston_quantiles import QuantileConfig, ORIGINAL_BOUNDS, heston_quantiles
from five_mode_transport.rank_profile import rank_profile, save_profile

REF = ORIGINAL_BOUNDS.mean(axis=1)

def make_basis():
    cfg = QuantileConfig(n_fft=1024, n_quantiles=64, n_bins=64)
    r = heston_quantiles(REF, cfg)
    params = [REF.copy() for _ in range(6)]
    for i, vec in enumerate(params):
        vec[i] += 0.02 * (ORIGINAL_BOUNDS[i,1] - ORIGINAL_BOUNDS[i,0])
    quantiles = np.vstack([heston_quantiles(p, cfg).quantiles for p in params])
    modes, _, _ = fit_reference_modes(quantiles, r.quantiles, r.u, 5)
    from five_mode_transport.heston_quantiles import grid_weights
    return {
        "modes": modes, "weights": grid_weights(r.u),
        "reference_quantiles": r.quantiles, "reference_eta": REF,
        "bounds": ORIGINAL_BOUNDS, "s0": np.array(cfg.s0),
        "maturity": np.array(cfg.maturity), "n_fft": np.array(cfg.n_fft),
        "n_quantiles": np.array(cfg.n_quantiles), "n_bins": np.array(cfg.n_bins),
        "x_half_width": np.array(cfg.x_half_width),
        "min_retained_mass": np.array(cfg.min_retained_mass),
        "max_negative_mass": np.array(cfg.max_negative_mass),
        "u_grid": r.u, "bin_edges": r.bin_edges
    }

def test_reference_zero_modes_and_threshold_reporting():
    basis = make_basis()
    result = rank_profile(REF, basis, thresholds=(0.9, 0.999), max_modes=5)
    assert len(result["ranks"]) == 6
    assert np.allclose(result["coefficients"], 0, atol=1e-8)
    assert result["ranks"][0]["n_modes"] == 0
    for value in result["report"]["thresholds"].values():
        if value["attained"]:
            assert value["minimum_modes"] is not None
        else:
            assert value["minimum_modes"] is None

def test_common_basis_and_extrapolation(tmp_path):
    basis = make_basis()
    target = REF.copy()
    target[0] += 0.01
    result = rank_profile(target, basis, thresholds=(0.999999999,), max_modes=4)
    assert len(result["coefficients"]) == 4
    assert result["report"]["within_fitted_parameter_bounds"]
    assert result["ranks"][0]["fraction_displacement_energy_captured"] <= 1.
    save_profile(result, tmp_path)
    assert (tmp_path / "rank_profile.csv").exists()
    outside = REF.copy()
    outside[0] = ORIGINAL_BOUNDS[0,1] + 0.001
    out = rank_profile(outside, basis, thresholds=(0.99,))
    assert not out["report"]["within_fitted_parameter_bounds"]
    assert "EXTRAPOLATION" in out["report"]["warning"]
