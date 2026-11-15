from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET_PATH = REPO_ROOT / "saml-d.parquet"
OUTPUT_DIR = REPO_ROOT / "strategic_detector" / "outputs"

WINDOWS = [7, 14, 30]


@dataclass(frozen=True)
class EconomicParams:
    """Payoff parameters for the Yang, Yao & Zhou (2026, Management Science) signaling
    game, renamed onto AML transaction monitoring. There is no "correct" value for any
    of these -- the paper treats them as the designer's own modeling choice too. Pick
    illustrative numbers, then sensitivity-check them.

    rho            : prior P(transaction is legitimate)                          -- paper's rho
    delta_r_high   : bank's payoff from correctly clearing a legitimate txn       -- Delta^R_H
    delta_r_low    : bank's loss from a laundering txn getting through           -- Delta^R_L (> 0, a loss)
    delta_s_low    : launderer's payoff from passing a laundering txn as legit   -- Delta^S_L
    delta_s_high   : legitimate customer's payoff from being correctly cleared   -- Delta^S_H
    lying_cost     : launderer's cost of restructuring to mimic a legit pattern  -- C
    """

    rho: float
    delta_r_high: float
    delta_r_low: float
    delta_s_low: float
    delta_s_high: float
    lying_cost: float

    def __post_init__(self) -> None:
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must be in (0, 1)")
        if self.delta_r_high <= 0 or self.delta_r_low <= 0:
            raise ValueError("delta_r_high and delta_r_low must be positive")
        if self.delta_s_high <= 0 or self.delta_s_low <= 0:
            raise ValueError("delta_s_high and delta_s_low must be positive")
        if not (0.0 < self.lying_cost < self.delta_s_low):
            raise ValueError(
                "lying_cost (C) must satisfy 0 < C < delta_s_low (paper's Section 2.1, "
                "otherwise the low-type sender never lies and beta_hat/sigma_s_star are "
                "undefined or trivial)"
            )
        if self.lying_cost >= self.delta_s_high:
            warnings.warn(
                "the paper's blanket scope assumption is C < min(delta_s_high, delta_s_low); "
                "only C < delta_s_low is actually load-bearing for sigma_s_star/beta_hat here, "
                "and delta_s_high only feeds the high-type-sender payoff readout in Table A.1, "
                "so this is left as a warning rather than an error.",
                stacklevel=2,
            )
        if self.delta_s_low > self.delta_r_low:
            warnings.warn(
                "delta_s_low > delta_r_low violates the paper's assumption that successful "
                "lying by the low type does not raise welfare (Section 2.1). The equilibrium "
                "formulas still evaluate, but Propositions 3-5's optimality claims no longer apply.",
                stacklevel=2,
            )

    @property
    def beta_hat(self) -> float:
        """True-positive rate separating the two equilibrium regimes (Lemma 3 / Proposition 1)."""
        return 1.0 - self.lying_cost / self.delta_s_low
