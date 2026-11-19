from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_PARQUET_PATH


def load_transactions(parquet_path: Path = DEFAULT_PARQUET_PATH) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "day" not in df.columns:
        df["day"] = (
            (df["Timestamp"] - df["Timestamp"].min()).dt.total_seconds() // 86400
        ).astype("int32")
    return df


def distinct_receivers(df: pd.DataFrame, window_days: int) -> np.ndarray:
    """Distinct receivers of each sender within each window_days-day window, repeated on
    every payment that falls in it. Ported from the notebook's `distinct_receivers` helper."""
    key = pd.DataFrame(
        {
            "s": df["Sender_account"].to_numpy(),
            "w": (df["day"] // window_days).to_numpy(),
            "r": df["Receiver_account"].to_numpy(),
        }
    )
    return key.groupby(["s", "w"])["r"].transform("nunique").to_numpy()
