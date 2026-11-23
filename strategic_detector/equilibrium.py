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
    """
     - out of all laundering transactions, what fraction disguise themselves, 
    once everyone is behaving optimally given everyone else's behavior?
    - once things settle, what share of laundering payments are disguised as legitimate?

    Equilibrium probability that a laundering transaction mimics a legitimate one, for
    an exogenously given detector {beta, alpha} with 0 < alpha < beta <= 1

    At beta == beta_hat the equilibrium is a whole interval; this returns its
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
    '''
    - once equilibrium settles out, how well off is the bank, how well off are the launderers, 
    and how well off are the legitimate customers, on average?
    - once things settle, how well off is each party (the bank, the launderers and honest customers)?
    '''
    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    _check_domain(alpha, beta)

    high_tp = beta >= p.beta_hat

    with np.errstate(divide="ignore", invalid="ignore"):
        # beta > beta_hat; Expected utilities for the bank, Launderer, and honest customer
        eur_high = (1 - alpha / beta) * p.rho * p.delta_r_high
        eus_l_high = np.zeros_like(beta)
        eus_h_high = p.delta_s_high - (p.delta_s_low - p.lying_cost) * p.delta_s_high / p.delta_s_low * (
            alpha / beta
        )

        """
        - beta < beta_hat: Weak detector. separating-vs-pooling split depends on alpha relative to this cutoff.
        - alpha_cutoff -> highest false-alarm rate at which the bank can still afford to trust unflagged payments.
        """
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


if __name__ == '__main__':

    # python -m strategic_detector.equilibrium
    def describe(name, x):
        arr = np.asarray(x)
        print(f"  {name:<18} type={type(x).__name__:<8} shape={str(arr.shape):<8} "
              f"dtype={str(arr.dtype):<8} value={np.round(arr, 4).tolist()}")

    params = EconomicParams(
        rho=0.7,           # 70% of transactions are legitimate
        delta_r_high=0.5,  # bank's gain from clearing a legit txn
        delta_r_low=0.5,   # bank's loss from clearing a laundering txn
        delta_s_low=0.5,   # launderer's gain if a laundering txn gets through
        delta_s_high=1.0,  # legit customer's gain from being cleared
        lying_cost=0.3,    # launderer's cost of disguising
    )
    print("INPUT  params:", params)
    print(f"       beta_hat = 1 - C/delta_s_low = {params.beta_hat:.4f}\n")

    # Case 1: a single detector, given as plain Python floats.
    # Case 2: four detectors in a 1-D array, two on each side of beta_hat (0.4).
    # Case 3: a 2-D grid of detectors, to show the output keeps the input's shape.
    beta_grid, alpha_grid = np.meshgrid([0.3, 0.6, 0.9], [0.05, 0.1])
    cases = {
        "1) scalars": (0.6, 0.1),
        "2) 1-D arrays": (np.array([0.2, 0.35, 0.5, 0.8]), np.array([0.05, 0.1, 0.1, 0.2])),
        "3) 2-D grid": (beta_grid, alpha_grid),
    }

    for title, (beta_in, alpha_in) in cases.items():
        print(f"=== Case {title} ===")
        print("INPUTS")
        describe("beta", beta_in)
        describe("alpha", alpha_in)

        sigma = sigma_s_star(beta_in, alpha_in, params)
        payoffs = equilibrium_payoffs(beta_in, alpha_in, params)

        print("OUTPUT sigma_s_star(...)")
        describe("sigma", sigma)
        print(f"OUTPUT equilibrium_payoffs(...)  -> {type(payoffs).__name__} with keys {list(payoffs)}")
        for key, value in payoffs.items():
            describe(key, value)
        print()

    # Case 4: an invalid detector (alpha >= beta) is rejected by _check_domain.
    print("=== Case 4) invalid input, alpha >= beta ===")
    try:
        sigma_s_star(0.2, 0.3, params)
    except ValueError as err:
        print(f"  ValueError: {err}")
