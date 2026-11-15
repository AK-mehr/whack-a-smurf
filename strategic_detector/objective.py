from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EconomicParams
from .equilibrium import equilibrium_payoffs


def designer_objective(
    beta: np.ndarray,
    alpha: np.ndarray,
    p: EconomicParams,
    weight_receiver: float = 1.0,
    weight_high_type: float = 0.0,
    weight_low_type: float = 0.0,
) -> np.ndarray:
    """J(beta, alpha) = weight_receiver * EU^R + weight_high_type * rho * EU^S_H
    + weight_low_type * (1 - rho) * EU^S_L (Lemma 6). Defaults to maximizing the
    receiver's (bank's) payoff alone, i.e. Proposition 3's objective."""
    payoffs = equilibrium_payoffs(beta, alpha, p)
    return (
        weight_receiver * payoffs["receiver"]
        + weight_high_type * p.rho * payoffs["high_type_sender"]
        + weight_low_type * (1 - p.rho) * payoffs["low_type_sender"]
    )


def optimal_point(roc: pd.DataFrame, p: EconomicParams, **weights: float) -> pd.Series:
    """Pick the (threshold, alpha, beta) row of an ROC curve that maximizes the
    designer's objective, restricted to the paper's interesting region 0 < alpha < beta
    (an uninformative or perverse operating point is never optimal; Section 4.1).

    `roc` must have columns 'threshold', 'alpha', 'beta' (as saved by
    scripts/build_roc_curves.py). Extra keyword arguments are forwarded to
    designer_objective (weight_receiver, weight_high_type, weight_low_type).
    """
    valid = roc[(roc["alpha"] > 0) & (roc["alpha"] < roc["beta"])].copy()
    if valid.empty:
        raise ValueError("no (alpha, beta) point on this ROC curve satisfies 0 < alpha < beta")
    valid["objective"] = designer_objective(
        valid["beta"].to_numpy(), valid["alpha"].to_numpy(), p, **weights
    )
    return valid.loc[valid["objective"].idxmax()]
