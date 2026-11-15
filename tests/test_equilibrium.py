import numpy as np
import pytest

from strategic_detector.config import EconomicParams
from strategic_detector.equilibrium import equilibrium_payoffs, sigma_s_star

PARAMS = EconomicParams(
    rho=0.7,
    delta_r_high=0.5,
    delta_r_low=0.5,
    delta_s_low=0.5,
    delta_s_high=1.0,
    lying_cost=0.3,
)


def test_beta_hat_matches_formula():
    assert PARAMS.beta_hat == pytest.approx(1 - 0.3 / 0.5)


def test_invalid_lying_cost_rejected():
    with pytest.raises(ValueError):
        EconomicParams(
            rho=0.7, delta_r_high=0.5, delta_r_low=0.5, delta_s_low=0.5, delta_s_high=1.0, lying_cost=0.5
        )


def test_sigma_s_star_in_unit_interval():
    beta = np.linspace(0.11, 0.99, 50)
    alpha = beta * 0.1
    sigma = sigma_s_star(beta, alpha, PARAMS)
    assert np.all(sigma >= 0.0) and np.all(sigma <= 1.0)


def test_sigma_s_star_rejects_alpha_not_less_than_beta():
    with pytest.raises(ValueError):
        sigma_s_star(np.array([0.2]), np.array([0.3]), PARAMS)


def test_equilibrium_payoffs_shapes_match_input():
    beta = np.linspace(0.11, 0.99, 10)
    alpha = beta * 0.05
    payoffs = equilibrium_payoffs(beta, alpha, PARAMS)
    assert payoffs["receiver"].shape == beta.shape
    assert payoffs["low_type_sender"].shape == beta.shape
    assert payoffs["high_type_sender"].shape == beta.shape
