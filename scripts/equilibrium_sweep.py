"""Step 4-5: apply the paper's closed-form equilibrium (Proposition 1) along each
control's ROC curve and plot the probability of mimicry against beta, looking for the
nonmonotonic ("Figure 3") shape.

Requires scripts/build_roc_curves.py to have been run first.
Run from the repo root:  python scripts/equilibrium_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from strategic_detector.config import OUTPUT_DIR
from strategic_detector.equilibrium import sigma_s_star
from strategic_detector.reporting import plot_sigma_vs_beta
from strategic_detector.scenario import ILLUSTRATIVE_PARAMS, ROC_CURVE_FILES


def main() -> None:
    fig, ax = plt.subplots()

    for label, path in ROC_CURVE_FILES.items():
        if not path.exists():
            print(f"skipping {label}: {path} not found (run build_roc_curves.py first)")
            continue
        roc = pd.read_csv(path)
        valid = roc[(roc["alpha"] > 0) & (roc["alpha"] < roc["beta"])]
        if valid.empty:
            print(f"skipping {label}: no points with 0 < alpha < beta on this curve")
            continue
        sigma = sigma_s_star(valid["beta"].to_numpy(), valid["alpha"].to_numpy(), ILLUSTRATIVE_PARAMS)
        plot_sigma_vs_beta(valid["beta"].to_numpy(), sigma, ax=ax, label=label)

    ax.axvline(ILLUSTRATIVE_PARAMS.beta_hat, linestyle="--", color="grey", linewidth=1)
    ax.set_title("Equilibrium probability of mimicry vs detector's true-positive rate")
    out_path = OUTPUT_DIR / "equilibrium_sigma_vs_beta.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
