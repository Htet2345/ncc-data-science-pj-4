"""Load and validate the lesson's one-feature regression dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if set(frame.columns) != {"Hours", "Scores"}:
        raise ValueError("CSV must have exactly the columns Hours,Scores")
    if len(frame) < 10:
        raise ValueError("Use at least 10 observations for a train/test split")
    for column in ("Hours", "Scores"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["Hours", "Scores"]].to_numpy(dtype=float)).all():
        raise ValueError("Hours and Scores must be finite numbers without missing values")
    if not frame.Hours.between(0, 24).all():
        raise ValueError("Hours must be between 0 and 24")
    if not frame.Scores.between(0, 100).all():
        raise ValueError("Scores must be between 0 and 100")
    if frame.Hours.nunique() < 5:
        raise ValueError("Use at least five distinct study-hour values")
    if frame.Scores.nunique() < 2:
        raise ValueError("Scores must vary to learn a useful relationship")
    if frame.duplicated().any():
        raise ValueError("Remove exact duplicate observations before splitting")
    return frame[["Hours", "Scores"]]
