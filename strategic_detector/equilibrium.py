from __future__ import annotations

import numpy as np

from .config import EconomicParams


def _check_domain(alpha: np.ndarray, beta: np.ndarray) -> None:
    if np.any(alpha <= 0) or np.any(alpha >= beta):
        raise ValueError(
            "this formula is only defined for 0 < alpha < beta <= 1 (an uninformative or "
            "perverse detector, alpha >= beta, is never optimal -- Section 4.1 of the paper)"
        )


def sigma_s_star(beta: np.ndarray, alpha: np.ndarray, p: EconomicParams) -> np.ndarray:
    """Equilibrium probability that a laundering transaction mimics a legitimate one, for
    an exogenously given detector {beta, alpha} with 0 < alpha < beta <= 1
    (Yang, Yao & Zhou 2026, Proposition 1).

    At beta == beta_hat the paper's equilibrium is a whole interval; this returns its
    lower end (the boundary value of the beta > beta_hat branch), matching the downward
    jump the paper's own Figure 3 shows at that point -- it is not a continuity bug.

    Values are clipped to [0, 1]: for some parameter combinations the closed-form
    expression falls outside the unit interval, which means a semiseparating equilibrium
    of this type does not exist there (a different equilibrium regime applies instead,
    i.e. complete lying or complete truth-telling; see Proposition 1's full case list).
    """
    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    _check_domain(alpha, beta)

    with np.errstate(divide="ignore", invalid="ignore"):
        low_regime = (1 - alpha) * p.rho * p.delta_r_high / ((1 - beta) * (1 - p.rho) * p.delta_r_low)
        low_regime = np.minimum(low_regime, 1.0)
        high_regime = alpha * p.rho * p.delta_r_high / (beta * (1 - p.rho) * p.delta_r_low)

    sigma = np.where(beta < p.beta_hat, low_regime, high_regime)
    return np.clip(sigma, 0.0, 1.0)


def equilibrium_payoffs(beta: np.ndarray, alpha: np.ndarray, p: EconomicParams) -> dict[str, np.ndarray]:
    """Expected equilibrium payoffs of the receiver (bank), the low-type sender
    (launderer) and the high-type sender (legitimate customer) for a detector {beta,
    alpha}, under the paper's Pareto-optimal-equilibrium refinement (Table A.1).

    Returns a dict with keys 'receiver', 'low_type_sender', 'high_type_sender', each an
    array the same shape as beta/alpha.
    """
    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    _check_domain(alpha, beta)

    high_tp = beta >= p.beta_hat

    with np.errstate(divide="ignore", invalid="ignore"):
        # beta in [beta_hat, 1): mixed-strategy regime (Table A.1, left column collapsed).
        eur_high = (1 - alpha / beta) * p.rho * p.delta_r_high
        eus_l_high = np.zeros_like(beta)
        eus_h_high = p.delta_s_high - (p.delta_s_low - p.lying_cost) * p.delta_s_high / p.delta_s_low * (
            alpha / beta
        )

        # beta in (0, beta_hat): separating-vs-pooling split depends on alpha relative to this cutoff.
        alpha_cutoff = 1 - (1 - p.rho) * p.delta_r_low * (1 - beta) / (p.rho * p.delta_r_high)
        separating = alpha <= alpha_cutoff

        eur_low_sep = (1 - alpha) * p.rho * p.delta_r_high - (1 - beta) * (1 - p.rho) * p.delta_r_low
        eus_l_low_sep = (1 - beta) * p.delta_s_low - p.lying_cost
        eus_h_low_sep = (1 - alpha) * p.delta_s_high
        eus_h_low_pool = (p.lying_cost / p.delta_s_low) * p.delta_s_high * (1 - alpha) / (1 - beta)

        eur_low = np.where(separating, eur_low_sep, 0.0)
        eus_l_low = np.where(separating, eus_l_low_sep, 0.0)
        eus_h_low = np.where(separating, eus_h_low_sep, eus_h_low_pool)

        eur = np.where(high_tp, eur_high, eur_low)
        eus_l = np.where(high_tp, eus_l_high, eus_l_low)
        eus_h = np.where(high_tp, eus_h_high, eus_h_low)

    return {"receiver": eur, "low_type_sender": eus_l, "high_type_sender": eus_h}
