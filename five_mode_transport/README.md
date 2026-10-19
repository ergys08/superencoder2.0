# Reference-anchored quantile transport (without a neural network)

This self-contained experiment lives in `superencoder2.0/five_mode_transport/`.
It reproduces the **classical quantile transport** part of the earlier
[`ergys08/optimal-transport`](https://github.com/ergys08/optimal-transport)
experiment, then changes the experiment to ask a more specific question:

> Starting from the quantile function of **one fixed Heston reference model**,
> how accurately can 0, 1, 2, ..., 5, ... modes reconstruct target Heston
> states as we move farther away from that reference? How many modes do we
> need across a specified parameter box?

**No neural network. No quantum circuit.** For an arbitrary target `eta`, the
`fit_transport.py` oracle computes the actual Heston target distribution and
projects its displacement onto the learned basis. This proves an approximation
is *possible with these classical modes*, not that it can be implemented by a
low-cost quantum circuit or predicted without computing the target distribution.

## 1. Three functional modules

| File | Inputs | Returns |
|---|---|---|
| `heston_quantiles.py` | An existing `HestonModel`, `HestonParameters`, or six-vector | FFT-derived terminal log-price quantile function `Q_eta(u)`, bin probabilities and diagnostics |
| `discover_transport_modes.py` | A reference Heston model, chosen parameter bounds, evaluation budget | Reference-anchored orthonormal quantile modes, per-rank mode importance and suggested rank; fidelity vs parameter distance, plots and a portable `transport_basis.npz` artifact |
| `fit_transport.py` | A saved basis, arbitrary Heston parameters and optionally `--modes 5` | Fitted classical coefficients, reconstructed quantiles/probabilities and amplitude fidelity against true target; explicit in/out-of-bounds warning |

All command-line examples must be run **from the repository root**, with the
repo's requirements installed (`pip install -r requirements.txt`).

### A. Calculate quantiles for one Heston model

```bash
python -m five_mode_transport.heston_quantiles \
  --eta 0.05 0.04 2.0 0.04 0.30 -0.70 \
  --out five_mode_transport/results/example_quantiles.npz
```

### B. Discover modes across the **same original Heston parameter range**

```bash
# First check the workflow (small, non-research-grade dataset).
python -m five_mode_transport.discover_transport_modes --quick \
  --out five_mode_transport/results/smoke

# Reproducible full study with reference fixed at the original box midpoint.
python -m five_mode_transport.discover_transport_modes \
  --n-train 256 --n-validation 96 --n-test 128 \
  --max-modes 12 --threshold 0.9999 --criterion worst \
  --out five_mode_transport/results/main
```

Original parameter ranges, **unchanged**:

| Parameter | Minimum | Maximum |
|---|---:|---:|
| `mu` | -0.03 | 0.12 |
| `v0` | 0.01 | 0.10 |
| `kappa` | 0.8 | 4.0 |
| `theta` | 0.02 | 0.10 |
| `xi` | 0.15 | 0.75 |
| `rho` | -0.9 | -0.1 |

To reuse the actual 2,100 distributions and 70/15/15 splits from the original
experiment rather than regenerate them, use:

```bash
python -m five_mode_transport.discover_transport_modes \
  --historical-dataset ../optimal-transport/results/dataset.npz \
  --out five_mode_transport/results/original_2100
```

This option requires the generated NPZ in the adjacent repository; it was not
committed to GitHub. The numerical grids must match.

The reference defaults to the midpoint vector
`(0.045, 0.055, 2.4, 0.06, 0.45, -0.5)`.
`S0=100`, `T=1 year`, `n_fft=4096`, `n_quantiles=512`, `n_bins=256`, and
`x_half_width=2.2` also match the original experiment.

If you want another reference or a narrower region:

```bash
python -m five_mode_transport.discover_transport_modes \
  --reference 0.05 0.04 2.0 0.04 0.30 -0.70 \
  --bounds-json five_mode_transport/example_bounds.json \
  --out five_mode_transport/results/custom
```

The optional JSON bounds file can be an array of six `[lower,upper]` pairs
in the parameter order above, or a dictionary with these six parameter names.
All lower/upper endpoints must be valid Heston parameters, and the reference
must be inside the box. The original ranges are used by default.

### C. Fit *any* Heston target to the saved basis

```bash
# Uses the rank recommended in the saved basis:
python -m five_mode_transport.fit_transport \
  --basis five_mode_transport/results/main/transport_basis.npz \
  --eta 0.05 0.04 2.0 0.04 0.30 -0.70 \
  --out five_mode_transport/results/fitted_model

# Force precisely five modes, regardless of the recommendation:
python -m five_mode_transport.fit_transport \
  --basis five_mode_transport/results/main/transport_basis.npz \
  --eta 0.05 0.04 2.0 0.04 0.30 -0.70 \
  --modes 5 --out five_mode_transport/results/fitted_five_modes
```

`fit_report.json` gives the fidelity, fitted coefficients, normalised distance
from reference and explicit `within_fitted_parameter_bounds` flag. It will
still fit a parameter set *outside* the box, but reports which components
exceed the bounds and marks the result `EXTRAPOLATION`.
`fit_states.npz` contains the true and fitted quantiles, classical
probabilities and positive-real amplitude vectors for further study. It also
returns `transport_source_log_prices`, `transport_target_log_prices`, and their
difference on the common `u` grid: an explicit sampled 1D transport map.

## 2. Mathematics — what is actually being fitted?

We compute the terminal log-price CDF by numerical inversion of the analytical
Heston characteristic function, and its quantile function

\[Q_\eta(u)=F_\eta^{-1}(u).\]

Fix the *actual*, not averaged, reference distribution `Q_ref`. For all
training samples compute their displacements

\[D_i(u)=Q_{\eta_i}(u)-Q_{\mathrm{ref}}(u).\]

The Chebyshev grid concentrates points in the tails. Weighted quadrature
`w_j` is essential: the quantity

\[\|D\|_w^2=\sum_j w_jD(u_j)^2\]

approximates squared 1D Wasserstein distance on the truncated quantile grid.
Take the **uncentred** singular value decomposition

\[D_{ij}\sqrt{w_j}=\sum_k U_{ik}\sigma_kV_{kj}.\]

This yields weighted-orthonormal modes

\[b_k(u_j)=V_{kj}/\sqrt{w_j}.\]

For an arbitrary new target, fit its coefficients using the **true target**
quantile function:

\[c_k(\eta)=\sum_j w_j\big(Q_\eta(u_j)-Q_{\mathrm{ref}}(u_j)\big)b_k(u_j).\]

Then approximate

\[\widehat Q^{(K)}_\eta=Q_{\mathrm{ref}}+\sum_{k=1}^{K}c_k(\eta)b_k.\]

This representation has an exact **zero-displacement reference anchor**:
`c(eta_ref)=0`, ignoring numerical integration precision. Importantly, this is
NOT identical to the **mean-centred** PCA in the original `optimal-transport`
repository, so the old five-mode result `~0.999818` does **not automatically**
apply here. The new experiment measures it independently.

A linear combination of quantile modes may not be monotone, so the code
applies weighted isotonic regression as necessary before converting it to a
256-bin distribution. The reported amplitude fidelity is

\[F=\left(\sum_j\sqrt{p_j\hat p_j}\right)^2.\]

Even with *every* quantile mode, a small bin-reconstruction discrepancy can
remain due to the finite `u` grid and tail extrapolation. Keep the 0-mode
reference reconstruction and other ranks as numerical controls.

## 3. Distance-from-reference experiment

We scale each parameter by the width of **your specified region**:

\[r(\eta)=\max_i\frac{|\eta_i-\eta_{\mathrm{ref},i}|}
                              {\eta_{i,\max}-\eta_{i,\min}}.\]

At each `--radii` level, sample `--n-per-radius` target models **exactly on
that shell while remaining inside the parameter box**. Their true Heston
quantiles are projected on 0, 1, 2, 3, 4, 5, selected, and maximum fitted modes.
This yields `fidelity_vs_radius.png` (mean and worst) and per-target CSV data.
Note that these samples are not necessarily uniform in angle or Wasserstein
distance from the reference, only in the declared normalised parameter radius.

All 64 parameter-box corners are explicitly evaluated as a separate stress test.
**These finite samples cannot certify fidelity over the entire continuous box.**
The optional conservative rank-selection rule `--criterion worst` selects the
smallest rank whose *minimum* over validation samples plus the 64 corners meets
`--threshold`. An alternative `--criterion p05` uses their fifth percentile.
If no fitted rank passes, the program reports `passed=false` and returns
`max_modes` as a **non-certified fallback**, rather than claiming success.
The separate test set is not used in rank selection.

## 4. Outputs and interpretation

- `transport_basis.npz`: reusable fitted basis, weights, Heston reference,
  parameter box, grid settings and selected rank. No pickle required.
- `summary.json`: exact settings, ranks, feasibility, validation/test metrics,
  and explicit limitations.
- `mode_significance.csv`: *incremental* displacement energy per mode,
  cumulative energy, improvement in validation average/worst fidelity and W2,
  plus 64-corner scores. **The fifth mode's value is measured, not assumed.**
- `test_by_rank.csv`: all ranks on one independent Heston test set.
- `fidelity_vs_radius.csv`: mean, 5th percentile and worst fidelity at each
  radius and selected rank, computed from fresh target Heston quantiles.
- `fidelity_vs_radius_samples.csv`: raw target parameters, errors and fidelities.
- `mode_significance.png`, `mode_functions.png`, `fidelity_vs_radius.png`: visual summaries.
- `transport_examples.npz`: an explicit quantile-pair transport from reference
  to one target at each radial shell (five modes unless fewer were fitted).
  At each quantile level `u`, the sampled transport pairs are
  `(Q_ref(u), Q_hat_target(u))`.
- `empirical_reach.json`: farthest **scanned** radius at which every preceding
  shell met the threshold, separately for each tested rank; not a guarantee.

Generate the complete data in one run, then open the plots and CSV files.

## 5. Scientific limitations

1. **This is a classical transport representation study.** It is not quantum
   transport by parameterised quantum gates; all fidelities are from classical
   overlap of amplitude vectors.
2. Per-target coefficient fitting needs the **exact/FFT Heston target** at
   evaluation time. A separate approach will be needed to obtain coefficients
   without computing the target when deploying a quantum circuit.
3. Fitting on a chosen parameter box means extrapolation has **no guarantee**.
   The fitting CLI checks bounds even if an out-of-range model is supplied.
4. The chosen rank is based on sampled/finitely enumerated validation cases,
   not a rigorous analytic bound over the continuous parameter space.
5. Modes and their signs depend on SVD convention, but the projected states do
   not. More modes do not necessarily improve **rebinned fidelity** monotonically
   because monotonicity correction and finite binning are nonlinear.
6. Fixed `S0` and maturity; only the terminal log-price distribution is
   encoded, not the full time series. Model parameters should share the same
   physical units as the original experiment.

## Shared SVD language: how many modes does any target need?

**New command:** `five_mode_transport.rank_profile`.

The discovery command already fits a **single common, weighted SVD basis** to
the sampled reference-relative displacements `Q_eta - Q_ref`, without mean
centring. It saves the ordered functions `b_1,...,b_K` to
`transport_basis.npz`. These functions are the SAME for every target fitted
to that saved file. The coefficients are target-specific and determined by
orthogonal projection against the exact target quantiles.

To study a particular target, evaluate EVERY prefix of that common basis:

```bash
python -m five_mode_transport.rank_profile \
  --basis five_mode_transport/results/main/transport_basis.npz \
  --eta 0.05 0.04 2.0 0.04 0.30 -0.70 \
  --thresholds 0.99 0.999 0.9999 \
  --out five_mode_transport/results/target_A
```

`rank_profile.json` gives a row for zero modes (identity), then 1 through the
number of modes saved in the basis. For each rank it reports amplitude fidelity,
projection residual, quantile error and whether monotonicity correction was
needed. `rank_profile.csv`, `fidelity_by_mode.png` and
`target_projection.npz` provide tabular, visual and numeric outputs.

A threshold's `minimum_modes` is the first *observed* prefix attaining it.
If no prefix passes, `minimum_modes` is `null`, rather than silently
pretending that the maximum rank was sufficient. Binned fidelity is not
necessarily monotone in rank.

To compare another target, use the **same** `--basis` path with a different
`--eta`. This is what makes mode counts meaningful *between targets*.

To change the reference, use `--reference` when running
`discover_transport_modes` and save a new basis. Different reference choices
produce different transport languages. `--bounds-json` determines where
the SVD is trained; it is not mathematically necessary for defining exact
quantile transport, but it is necessary to specify which displacements the
finite-dimensional modes should approximate well. Outside its fitted box,
the rank-profile report prints an EXTRAPOLATION warning but still measures
fidelity using the true Heston target.

**Interpretation warning:** These are target-specific *oracle* coefficients,
not a learned parameter-to-coefficient mapping, and not a quantum gate
implementation. One target-specific bespoke displacement would trivially be
one exact mode, so only compare modes drawn from the same precomputed basis.
A six-parameter manifold need not lie in a six-dimensional *fixed linear*
quantile space.
