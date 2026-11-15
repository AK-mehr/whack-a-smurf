"""Step 1-2: build the full ROC curve for each control (not just the notebook's four
budgets) and save them so the later scripts can reuse them without recomputing.

Run from the repo root:  python scripts/build_roc_curves.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from strategic_detector.config import OUTPUT_DIR, WINDOWS
from strategic_detector.data import distinct_receivers, load_transactions
from strategic_detector.detectors import control_a_roc, control_b_roc
from strategic_detector.reporting import plot_roc_curves


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txns = load_transactions()
    is_laundering = (txns["Is_laundering"] == 1).to_numpy()
    amount = txns["Amount"].to_numpy()

    curves: dict[str, tuple] = {}

    thresholds, alpha, beta = control_a_roc(amount, is_laundering)
    curves["Control A (amount)"] = (alpha, beta)
    pd.DataFrame({"threshold": thresholds, "alpha": alpha, "beta": beta}).to_csv(
        OUTPUT_DIR / "roc_control_a.csv", index=False
    )

    for w in WINDOWS:
        nr = distinct_receivers(txns, w)
        thresholds, alpha, beta = control_b_roc(nr, is_laundering)
        label = f"Control B ({w}d window)"
        curves[label] = (alpha, beta)
        pd.DataFrame({"threshold": thresholds, "alpha": alpha, "beta": beta}).to_csv(
            OUTPUT_DIR / f"roc_control_b_{w}d.csv", index=False
        )

    plot_roc_curves(curves)
    plt.title("ROC curves: amount threshold vs distinct-receiver-count controls")
    out_path = OUTPUT_DIR / "roc_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved ROC curves (CSV) and comparison plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
