# Transport Discovery — SuperEncoder 2.0

> **Question:** Can one inexpensive, reusable *parameter-dependent transport circuit* transform a **fixed Heston reference state** into every target state in our original six-parameter domain, without running new per-target gate optimisation at inference?
>
> Architecture = an ordered list of real-amplitude-preserving Pauli-rotation **primitives**, each with its own continuously adjustable angle. A small neural network maps the six Heston parameters to those angles.

## 0. What this is (and is not)

For fixed \(S_0=100\), maturity \(T=1\), and the Heston parameters \(\eta=(\mu,v_0,\kappa,\theta,\xi,\rho)\), we want

\[
P_{\mathrm{ref}}|0^N\rangle=|\psi_{\mathrm{ref}}\rangle,\qquad
T_{\mathcal A}(f_\phi(\eta))|\psi_{\mathrm{ref}}\rangle\approx|\psi_\eta\rangle.
\]

We **prepare the reference state with the same fixed circuit every time**. We search for a **cheap second circuit** that depends on the *target parameters*, but does not change its gate ordering/topology. The reference preparation cost is excluded from the transport-only gate budget; both costs should be added for end-to-end benchmarking.

The initial search is **not** forced to use the previously discovered five quantile modes. Those can be explored later as a *parameterisation* of the angle map or as additional screening tangents. The experiment directly measures six physical Heston parameter tangents at the fixed reference state.

**A learned candidate is not guaranteed to satisfy the requested fidelity threshold.** The code honestly reports `no_candidate_met_validation_threshold` if nothing does. No quantum advantage is claimed.

## 1. Reproduce the original Heston parameter domain

| Input | Min | Max |
|---|---:|---:|
| Drift `mu` | -0.03 | 0.12 |
| Initial variance `v0` | 0.01 | 0.10 |
| Mean reversion `kappa` | 0.8 | 4.0 |
| Long-run variance `theta` | 0.02 | 0.10 |
| Volatility of variance `xi` | 0.15 | 0.75 |
| Correlation `rho` | -0.9 | -0.1 |

The six midpoint values define the reference model. The reference is **not the average of the training states**, and we never search for a different reference to improve results. This removes a degree of freedom and keeps comparisons fair.

The default script samples fresh distributions using a seeded 6D Latin hypercube and the repository's `heston.py` characteristic-function inversion. It is not the identical historical 2,100-example dataset. **To use the exact historical 70/15/15 split**, pass the generated `optimal-transport/results/dataset.npz` (see commands). We use the *bin probability arrays* from that dataset directly for the targets and re-bin the fixed reference to its original log-price grid. This explicitly avoids a subtle mismatch between the two repositories' FFT/bin endpoints.

All state splits are **disjoint by Heston parameter vector**, never merely by angle evaluation or source–target pair. The validation set is used to select the winning neural architecture; the test split is read once after selection.

## 2. Set up & run

From the root of `superencoder2.0` after activating your virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q

# Smoke test: fast and deliberately too small for research conclusions
python "transport discovery/run.py" --quick

# Actual baseline search (192 train / 64 validation / 64 held-out states)
python "transport discovery/run.py" \
  --output "transport discovery/results/main"

# Exact saved historical Heston dataset and 1470/315/315 original splits:
python "transport discovery/run.py" \
  --dataset "../optimal-transport/results/dataset.npz" \
  --output "transport discovery/results/original"
```

When using a custom NPZ, its train/validation/test indices override the sample-count settings. All results land in ignored `results/` directories. Large experiments can take substantial CPU time: the oracle fits many circuits and the final network is trained separately for each finalist. The initial smoke test verifies the pipeline, **not** the fidelity hypothesis.

By default the search uses 8 qubits, 60 proposed architectures, 12 oracle-screened architectures, and 3 neural finalists. Edit `Settings` in `config.py` or change the CLI's `--max-cx` and `--n-candidates`.

## 3. Primitive grammar

A primitive is \(R_P(\theta)=\exp(-i\theta P/2)\), where \(P\) is a Pauli word such as `IZYIIIII` (q0 is the *first* letter). We consider contiguous 1-, 2-, 3- and 4-qubit Pauli words plus non-neighbour two-qubit words. Pauli strings contain an **odd number of `Y` operators**, which makes the resulting unitary real in this basis and preserves the target family's real amplitude subspace.

The standard synthesis is: rotate each non-identity Pauli to Z, compute the parity with CNOTs, apply `RZ(theta)`, undo parity, and undo basis changes. A weight-\(k\) Pauli rotation has a conservative pre-cancellation cost of \(2(k-1)\) CNOTs. The evaluator also counts *compiled* CNOTs and circuit depth with Qiskit using `['u','cx']` at representative nonzero angles. It reports depth `-1` as **unavailable** if Qiskit isn't installed; it never labels the conservative estimate as a transpilation result.

Each candidate is an **ordered list** of primitives, for example:

```json
{"words": ["YIIIIIII", "IYZIIIII", "IIIZYIII"]}
```

Its state is `U_L(theta_L) ... U_1(theta_1) |psi_ref>`. For **zero angles**, every primitive is identity, so \(T(\eta_{\mathrm{ref}})=I\) exactly.

This initial vocabulary is deliberately restricted for a controlled comparison. It does **not** claim to explore all transport algorithms, MPS isometries, arithmetic circuits, or ancillary/measurement-based approaches. Expansion to other primitives should include an independent simulator, a correct Qiskit decomposition, and resource tests.

## 4. Three levels of evaluation (and why we separate them)

**Stage A — Local tangent expressivity, no optimisation.**

Finite-difference the six Heston state tangents with respect to *normalised* model parameters at the reference state, giving \(W\in\mathbb R^{2^N\times6}\). For each primitive, calculate

\[
J_\ell=\left.\frac{\partial T(\boldsymbol\theta)|\psi_\mathrm{ref}\rangle}{\partial\theta_\ell}\right|_{\theta=0}= -\frac i2P_\ell|\psi_\mathrm{ref}\rangle.
\]

Compute the normalised projection error

\[
R_\mathrm{tangent}=\frac{\|W-JJ^+W\|_F}{\|W\|_F}.
\]

This is a **necessary first-order test for a smooth identity-anchored parameter map to reproduce the Heston family locally**, not a global fidelity guarantee. We rank many architectures cheaply with this residual plus a modest CNOT penalty. Note: identical primitives give the same first-order columns regardless of ordering; nonlinear expressivity is tested at the next stages.

**Stage B — Per-target numerical oracle, diagnostic ONLY.**

Take the top 12 from Stage A. On the **same 12 training states**, optimise each circuit's angles using L-BFGS-B, fixed iterations and restarts, maximising fidelity. The simulator computes exact analytical gradients. The returned value is the **best found under that finite budget**, never a certified maximum. It tests whether local expressivity extrapolates to finite changes and is not available at deployment. Prefer strong oracle fidelity and low CNOT count when selecting finalists.

**Stage C — Learn a shared angle map, then test once.**

For 3 finalists, train the same two-hidden-layer `tanh` MLP end-to-end on training-state fidelity. It predicts one angle per primitive. Angles are anchored to zero at the Heston reference by

\[
\theta(\eta)=\pi\,[\tanh f_\phi(x_\eta)-\tanh f_\phi(0)],\quad
x_\eta=(\eta-\eta_\mathrm{ref})/(\eta_\mathrm{max}-\eta_\mathrm{min}).
\]

*No per-target angle optimisation is used for validation/test.* Choose the cheapest architecture satisfying the **validation fifth-percentile fidelity** threshold (default `0.9999`). If none qualifies, select the best validation fifth-percentile architecture **and mark the run as failure to meet the target**. Finally evaluate that one winner on the independent test split.

Saved metrics: mean, fifth percentile, worst fidelity, fraction above 0.999 / 0.9999, identity baseline and per-state predicted angles; compiled CNOT/depth and initial reference-preparation baseline when Qiskit is available.

## 5. Candidate evolution vs actual LLM evolution

`run.py` does **local, deterministic seeded evolutionary mutation**, not secretly an LLM call. It starts from multiple primitive families, then adds/replaces/removes/reorders primitives. This provides a crucial **non-LLM control experiment**.

To try **LLM-generated candidate architectures**, there are two supported paths:

**Simple proposal list:** Ask an LLM to produce `[{"words":[...]}, ...]`, then pass `--proposals path/to/proposals.json`. The same evaluator handles them. This does not require an external API.

**OpenEvolve adapter:** Install and configure [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) and a model provider according to its documentation, then, from the repository root:

```bash
openevolve-run "transport discovery/llm_candidate.py" \
  "transport discovery/openevolve_evaluator.py" --iterations 40
```

OpenEvolve edits a *Python program containing a literal `WORDS = [...]`*. Our `openevolve_evaluator.py` **parses but never executes that code**, verifies all primitives against an allow-list, then evaluates on a *fixed TRAINING subset only*. It returns `combined_score`, `oracle_fidelity`, `tangent_coverage`, `cx_upper_bound` and `number_of_gates`. This is suitable for LLM-guided proposal exploration, not a substitute for the full NN/held-out evaluation in `run.py`. Copy the most promising `WORDS` lists into a `proposals.json` and run the main pipeline again. Do not let LLM-generated code modify the evaluator or split definitions.

**Note:** OpenEvolve is optional and **not installed by `requirements.txt`**. The native seeded search runs without an API key. This repository does not pretend to launch paid model calls automatically.

## 6. Output files and interpretation

- `settings.json`, `domain.json`: every choice, seed, bounds, FFT resolution, data source.
- `reference.npz`: immutable reference state and parameters.
- `screening.csv`: *all* submitted architectures and tangent-score/costs.
- `oracle.csv`: finite-budget per-target optimiser statistics.
- `validation.csv`: shared-NN results for finalists; **used to select winner**.
- `winner.json`: topology of winning transport circuit.
- `winner_model/`: saved neural weights and training curve.
- `heldout_predictions.npz`: final target/predicted states and gate angles.
- `summary.json`: winner, status, held-out test metrics, resource costs and limitations.

A strong result must beat **identity** (otherwise near-reference states are trivial), meet the full validation/test fidelity criteria, AND demonstrate cost savings compared with *direct preparation* when considering the fixed reference-prep overhead. A low CNOT count alone isn't useful if the reference + transport requires more gates than direct preparation.

## 7. Limitations and next comparisons

1. Fidelity uses *ideal noiseless statevectors*, not physical gate errors or noisy Heston discretisation.
2. Architecture search is restricted to real Pauli rotations, with no ancillas, no mid-circuit measurements, and no arbitrarily generated new primitives.
3. The tangent screen is only local. Very shallow circuits with tangent span of dimension 6 can still fail on distant family states.
4. Per-target oracle optimisation is finite-budget and is **never** treated as a proof of expressivity.
5. This uses an eight-qubit terminal **distribution**, not a full Heston time-series encoding.
6. Large resource savings remain a hypothesis, **not an established result**.
7. Need to benchmark against constructive `StatePreparation`, conditional-rotation trees, and deterministic MPS compilation at matched fidelities. In the reported `summary.json`, generic state preparation is measured on the fixed reference and one validation target (when Qiskit is installed) as an initial sanity comparison, not a robust benchmark over the family.
8. The ability to predict angles by a NN is investigated empirically, not inferred from a high oracle fidelity.

### Map of the source code

| File | Responsibility |
|---|---|
| `config.py` | Original Heston parameter domain, fixed reference, reproducible budgets |
| `data.py` | Heston distributions, optional historical NPZ, disjoint state splits |
| `primitives.py` | Gates, exact real simulator, Qiskit synthesis and CX cost |
| `candidate.py` | Candidate architecture grammar and primitive mutations |
| `evaluator.py` | Tangent coverage, exact oracle-gradient, statistical metrics |
| `learner.py` | Shared parameter-to-angle NN trained on fidelity |
| `run.py` | Three-stage search, selection, final test, outputs |
| `llm_candidate.py` | Safe editable OpenEvolve template |
| `openevolve_evaluator.py` | LLM-independent allowed-primitive evaluator |
| `../tests/test_transport_discovery.py` | Mathematical/unit/regression checks |

## 8. Use the winning circuit for a new Heston model

After `run.py` has produced `winner_model/`, request angles using **only** the Heston parameters (no target-specific Fourier inversion and no optimiser):

```bash
python "transport discovery/inference.py" \
  --run "transport discovery/results/main" \
  --eta 0.05 0.04 2.0 0.04 0.30 -0.70

# If Qiskit is installed, also export the transport ONLY (not P_ref) as QPY:
python "transport discovery/inference.py" \
  --run "transport discovery/results/main" \
  --eta 0.05 0.04 2.0 0.04 0.30 -0.70 \
  --qpy "transport discovery/results/main/transport.qpy"
```

The returned angles define a circuit to apply **after** the fixed reference state has been prepared. `--qpy` exports the circuit in Qiskit's binary format; it is deliberately ignored by Git if stored inside `results/`.

## 9. Robust family-wide objective (October 2026 update)

**The goal is *one fixed reference preparation* followed by *one fixed transport
architecture* whose angles depend on any target Heston parameter tuple.** The
parameters remain exactly as defined in Section 1. Unlike the initial pilot,
we now explicitly optimise **poorly served regions of the entire admissible
six-dimensional box**, not just average fidelity on random models.

Let the frozen transport map be

\[
F_\phi(\eta;\mathcal A)
=|\langle\psi_\eta|T_{\mathcal A}(f_\phi(\eta))|\psi_{\rm ref}\rangle|^2.
\]

The ideal target is

\[
\max_{\mathcal A,\phi}\;\inf_{\eta\in\Omega} F_\phi(\eta;\mathcal A)
\quad\text{subject to a compiled CNOT budget.}
\]

**This implementation only APPROXIMATES that minimax objective.** It is not
a proof of universal fidelity. The following sequence is used:

1. **Candidate search:** reference-state tangent coverage and a CX penalty
   screen many topologies. The finite-budget *oracle* now includes the same
   fixed difficult domain-corner targets for every architecture and ranks by
   the **lowest per-target oracle fidelity**, not the average. Oracle angles
   are a diagnostic only and never used at deployment.
2. **Shared NN training:** optimise a tail-sensitive objective
   \((1-\lambda)\operatorname{mean}(1-F)+\lambda\operatorname{CVaR}_{q}(1-F)\),
   with \(q=25\%\) and \(\lambda=0.75\) by default. The CVaR approximation
   averages the worst-loss examples in each minibatch. This prioritises hard
   targets but is not a global worst-case guarantee.
3. **Finalist selection (robustness validation):** after training, evaluate
   all **64 six-dimensional parameter-box corners**, another **128 independent
   interior samples**, and an adversarial **differential-evolution** search
   that tries to find parameters of lowest fidelity. Require the worst
   **observed** fidelity and the ordinary validation 5th percentile to exceed
   the user-chosen threshold (default 0.9999). Of feasible architectures,
   select the cheapest compiled CX count. If none passes, report FAILURE and
   select the architecture with highest observed minimum for diagnosis.
4. **Final test/audit:** freeze the architecture and NN and only then evaluate
   the original held-out test states plus all 64 corners, **512 fresh interior
   samples** and another independently seeded adversarial search. These
   final results NEVER feed back into model selection.

The 64 *public deterministic corners* necessarily recur in validation and
final audit. Only the interior/adversarial random seeds are independent. The
adversarial optimiser is a numerical stress test, not a certification of the
continuous minimum. New risk measures and sample budgets are in `config.py`.
The `--quick` run uses much smaller budgets and is NOT evidence of coverage.

Outputs include `finalist_#_robust_validation.npz`,
`robustness_audit.json`, `robustness_audit.npz`. Every adversarial worst-case
parameter vector, fidelity, evaluation count and source is recorded. The
`summary.json` status is `robust_threshold_not_met` unless validation and
final audit both meet their thresholds. The Heston state encoding, including
all corners and adversarial states, uses the original saved NPZ's log-price
bin edges when `--dataset` is supplied (no mixing of bin conventions).

### Robustness CLI

```bash
python "transport discovery/run.py" --output "transport discovery/results/robust" \
  --robust-validation-samples 256 --robust-test-samples 1024 \
  --adversarial-iters 30
```

More samples and more adversarial iterations give a stronger **empirical**
challenge, at additional CPU expense. Even exhaustive testing of the 64
vertices does not bound fidelity inside the parameter box.

## 10. Run an actual LLM with OpenEvolve

The default `run.py` still performs **NON-LLM evolutionary mutations**. An
actual LLM run requires the separately installed OpenEvolve package and a
provider API key. This project includes working provider templates based on
OpenEvolve's public configuration (`openevolve_gemini.yaml` and
`openevolve_openai.yaml`). In both cases **API requests may incur charges**.
A ChatGPT subscription does not provide OpenAI API credits.

From the `superencoder2.0` root in your activated virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pip install openevolve
```

**Option A — Gemini** (get an API key at https://aistudio.google.com/apikey):

```bash
# Type your own API key; do NOT paste it into a shared repo or commit it.
read -s GEMINI_API_KEY; echo
export GEMINI_API_KEY

openevolve-run "transport discovery/llm_candidate.py" \
  "transport discovery/openevolve_evaluator.py" \
  --config "transport discovery/openevolve_gemini.yaml" \
  --iterations 40 --output "transport discovery/results/llm"
```

**Option B — OpenAI API** (create an API key at https://platform.openai.com/api-keys):

```bash
read -s OPENAI_API_KEY; echo
export OPENAI_API_KEY

openevolve-run "transport discovery/llm_candidate.py" \
  "transport discovery/openevolve_evaluator.py" \
  --config "transport discovery/openevolve_openai.yaml" \
  --iterations 40 --output "transport discovery/results/llm"
```

Use your shell prompt to enter the key after `read -s` (input is hidden).
You can change model names/iterations in your local YAML file. The two
providers are mutually exclusive alternatives: choose one. For a small
paid-model test, use `--iterations 5` initially.

**Important search limitations:** OpenEvolve changes the *literal ordered
Pauli-word list*, meaning primitive type, location, ordering and count.
The LLM does **not** yet invent new gate decompositions or arbitrary
transport algorithms in this controlled search grammar. The independent
`openevolve_evaluator.py` never executes untrusted generated Python: it
parses `WORDS` with `ast.literal_eval`. Its ranking score is based on the
**worst finite-budget per-target oracle fidelity** on six ordinary training
states plus six publicly prescribed difficult corners, with a tangent/CX
penalty. This provides inexpensive architecture guidance, **not** a direct
measure of the learned map's worst-case fidelity. The robust trained-NN
comparison is performed in the next stage.

**Extract distinct architectures from OpenEvolve's checkpoints** (including
intermediate programs; this is more useful than taking just the last winner):

```bash
python "transport discovery/extract_proposals.py" \
  --checkpoints "transport discovery/results/llm/checkpoints" \
  --out "transport discovery/results/llm/proposals.json"
```

**Fit shared NN angle maps and test these LLM proposals** (no random mutants):

```bash
python "transport discovery/run.py" \
  --proposals "transport discovery/results/llm/proposals.json" \
  --proposal-only \
  --output "transport discovery/results/llm_verified"
```

To compare fairly against non-LLM evolutionary candidates, run the main
`run.py` without `--proposals`, using comparable numbers of oracle-screened
and NN-finalist architectures. The LLM evaluator's scores alone cannot be
compared directly with final NN-based transport fidelity.

The saved files `validation.csv`, `robustness_audit.json` and `summary.json`
are the most important results. Treat an LLM architecture that fails the
observed worst-case threshold as a **negative result**, not a successful
family-wide transport circuit.