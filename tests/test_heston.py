import numpy as np
import pytest

from heston import HestonModel, HestonParameters


def test_cf_normalised_and_hermitian():
    model = HestonModel(HestonParameters())
    phi = model.characteristic_function(np.array([0.0, 0.7, -0.7]))
    assert np.allclose(phi[0], 1)
    assert np.allclose(phi[2], np.conj(phi[1]))


def test_amplitude_normalisation():
    law = HestonModel(HestonParameters()).terminal_distribution(n_qubits=5, n_fft=2048)
    assert law.probabilities.shape == (32,)
    assert np.all(law.probabilities >= 0)
    assert np.isclose(law.probabilities.sum(), 1)
    assert np.isclose(np.sum(law.amplitudes ** 2), 1)


def test_mu_shifts_log_price_mean():
    model = HestonModel(HestonParameters())
    old = model.terminal_distribution(n_qubits=8)
    new = model.with_parameters(mu=.09).terminal_distribution(n_qubits=8)
    old_mean = np.sum(old.probabilities * np.log(old.price_midpoints))
    new_mean = np.sum(new.probabilities * np.log(new.price_midpoints))
    assert abs((new_mean - old_mean) - .04) < .002


def test_updated_model_is_independent():
    model = HestonModel(HestonParameters())
    other = model.with_parameters(rho=-.4)
    assert model.params.rho == -.7
    assert other.params.rho == -.4


def test_path_shapes():
    s, v = HestonModel(HestonParameters()).simulate_paths(10, 12, seed=7)
    assert s.shape == v.shape == (10, 13)
    assert np.allclose(s[:, 0], 100)


def test_bad_rho_rejected():
    with pytest.raises(ValueError):
        HestonParameters(rho=1.1)
