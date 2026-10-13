# SuperEncoder 2.0

Research code for neural quantum state preparation and Heston-model target states.

## Clone into your Applied Quantum Software Project folder (macOS)

```bash
cd "$HOME/Applied Quantum Software Project"
git clone https://github.com/ergys08/superencoder2.0.git
cd superencoder2.0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python heston.py
```

If your parent directory is elsewhere, substitute its actual path. In VS Code choose **Python: Select Interpreter** and select `.venv/bin/python`.

For Windows PowerShell: `py -m venv .venv`, `.venv\Scripts\Activate.ps1`, then `python -m pip install -r requirements.txt`.

## Reuse the Heston model in your research

```python
from heston import HestonModel, HestonParameters

params = HestonParameters(s0=100, mu=0.05, v0=0.04,
                          kappa=2, theta=0.04, xi=0.3, rho=-0.7, maturity=1)
model = HestonModel(params)
distribution = model.terminal_distribution(n_qubits=8)
p = distribution.probabilities           # 256 log-price bin probabilities, sums to 1
psi = distribution.amplitudes            # sqrt(p), normalized 8-qubit statevector
prices = distribution.price_midpoints    # terminal asset prices
new_model = model.with_parameters(rho=-0.5, xi=0.4)
new_psi = new_model.amplitude_state(n_qubits=8)
S, v = model.simulate_paths(n_paths=1000, n_steps=252, seed=7)
```

The Heston terminal distribution is computed with numerical Fourier inversion of the analytic characteristic function, not a noisy Monte Carlo histogram. It represents a **terminal price marginal**, not an entire multitime trajectory. Increase `n_fft` and `log_price_half_width` if numerical errors are a concern. The amplitude vector is a *target* state and does not itself supply a quantum circuit.

`mu` is the asset drift. Under a risk-neutral measure, set it to the risk-neutral drift if appropriate. The Feller condition is reported but not imposed. Simulated paths use approximate full-truncation Euler.
