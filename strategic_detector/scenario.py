from __future__ import annotations

from pathlib import Path

from .config import EconomicParams, OUTPUT_DIR, WINDOWS

# ILLUSTRATIVE economic parameters -- replace with your own business numbers (see README,
# step 3). These only need to be directionally sensible; the point of the scripts is to
# let you re-run everything cheaply once you have better numbers.
ILLUSTRATIVE_PARAMS = EconomicParams(
    rho=0.99896,  # 1 - the dataset's actual laundering rate (~0.104%)
    delta_r_high=5.0,  # bank's margin on a correctly cleared legitimate payment
    delta_r_low=5_000.0,  # bank's assumed loss (fines, remediation) from a missed laundering txn
    delta_s_low=20_000.0,  # value a launderer gets from passing a txn as legitimate
    delta_s_high=5.0,  # legitimate customer's benefit from being correctly cleared
    lying_cost=500.0,  # launderer's cost of restructuring (cf. the notebook's per-piece/per-receiver cost)
)

ROC_CURVE_FILES: dict[str, Path] = {
    "Control A (amount)": OUTPUT_DIR / "roc_control_a.csv",
    **{f"Control B ({w}d window)": OUTPUT_DIR / f"roc_control_b_{w}d.csv" for w in WINDOWS},
}
