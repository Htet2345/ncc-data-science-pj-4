import numpy as np
import pandas as pd
import pytest

from score_prediction.data import load_data
from score_prediction.model import DEFAULT_DATA


def test_sample_dataset():
    frame = load_data(DEFAULT_DATA)
    assert frame.shape == (32, 2)
    assert frame.Hours.min() == 0.5
    assert frame.Hours.max() == 9.8


@pytest.mark.parametrize("kind", ["missing_column", "extra_column", "few_rows", "nan", "inf", "text", "negative_hours", "over_24_hours", "negative_score", "over_100_score", "constant_hours", "constant_scores", "duplicates"])
def test_invalid_training_data_is_rejected(tmp_path, kind):
    frame = pd.read_csv(DEFAULT_DATA)
    if kind == "missing_column":
        frame = frame.drop(columns="Scores")
    elif kind == "extra_column":
        frame["Student"] = "demo"
    elif kind == "few_rows":
        frame = frame.head(5)
    elif kind == "nan":
        frame.loc[0, "Hours"] = np.nan
    elif kind == "inf":
        frame.loc[0, "Hours"] = np.inf
    elif kind == "text":
        frame["Hours"] = frame.Hours.astype(str)
        frame.loc[0, "Hours"] = "invalid"
    elif kind == "negative_hours":
        frame.loc[0, "Hours"] = -1
    elif kind == "over_24_hours":
        frame.loc[0, "Hours"] = 25
    elif kind == "negative_score":
        frame.loc[0, "Scores"] = -1
    elif kind == "over_100_score":
        frame.loc[0, "Scores"] = 101
    elif kind == "constant_hours":
        frame["Hours"] = 2
    elif kind == "constant_scores":
        frame["Scores"] = 50
    elif kind == "duplicates":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)
