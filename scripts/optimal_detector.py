"""Step 6-7: for each control's ROC curve, find the beta* that maximizes the designer's
objective (default: the bank's expected payoff alone, Proposition 3) and report the
concrete threshold (the Amount cutoff or the receiver-count K) that achieves it.

Requires scripts/build_roc_curves.py to have been run first.
Run from the repo root:  python scripts/optimal_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from strategic_detector.objective import optimal_point
from strategic_detector.scenario import ILLUSTRATIVE_PARAMS, ROC_CURVE_FILES


def main() -> None:
    for label, path in ROC_CURVE_FILES.items():
        if not path.exists():
            print(f"skipping {label}: {path} not found (run build_roc_curves.py first)")
            continue
        roc = pd.read_csv(path)
        try:
            best = optimal_point(roc, ILLUSTRATIVE_PARAMS, weight_receiver=1.0)
        except ValueError as exc:
            print(f"skipping {label}: {exc}")
            continue
        print(
            f"{label}: beta*={best['beta']:.4f}  alpha*={best['alpha']:.4f}  "
            f"threshold={best['threshold']:.4g}  J={best['objective']:.4f}"
        )


if __name__ == "__main__":
    main()
