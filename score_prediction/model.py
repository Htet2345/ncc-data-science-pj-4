"""Train on a reproducible split, evaluate held-out data and predict scores."""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from score_prediction.data import load_data

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "sample_data" / "student_scores.csv"
DEFAULT_OUTPUT = ROOT / "artifacts"


def train(data_path: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    frame = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[["Hours"]], frame["Scores"], test_size=0.2, random_state=42
    )
    if x_train.Hours.nunique() < 2:
        raise ValueError("Training split needs varying study hours")
    model = LinearRegression().fit(x_train, y_train)
    predicted = model.predict(x_test)
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train).predict(x_test)
    metrics = {
        "dataset": str(data_path.resolve()),
        "data_note": "Bundled data is synthetic; custom data must be described separately.",
        "rows": len(frame), "train_rows": len(x_train), "test_rows": len(x_test),
        "test_fraction": 0.2, "random_state": 42,
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(root_mean_squared_error(y_test, predicted)),
        "r2": float(r2_score(y_test, predicted)) if y_test.nunique() > 1 else None,
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "coefficient": float(model.coef_[0]), "intercept": float(model.intercept_),
        "training_hours_min": float(x_train.Hours.min()),
        "training_hours_max": float(x_train.Hours.max()),
        "train_row_indices": [int(index) for index in x_train.index],
        "test_row_indices": [int(index) for index in x_test.index],
        "scikit_learn_version": sklearn.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # This artifact contains the training-fitted model, not a fit on test rows.
    bundle = {"model": model, "metrics": metrics, "schema_version": 1}
    joblib.dump(bundle, output_dir / "model.joblib")
    results = pd.DataFrame({
        "row_index": x_test.index, "Hours": x_test.Hours.to_numpy(),
        "Actual": y_test.to_numpy(), "Predicted": predicted,
        "Residual": y_test.to_numpy() - predicted,
    })
    results.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from score_prediction.plots import save_plots
    save_plots(x_train, x_test, y_train, y_test, predicted, model, output_dir)
    return metrics


def predict(hours: float, model_path: Path = DEFAULT_OUTPUT / "model.joblib") -> dict:
    if not math.isfinite(hours) or not 0 <= hours <= 24:
        raise ValueError("Study hours must be a finite number between 0 and 24")
    # Only load artifacts you created or trust: joblib uses Python pickle.
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model artifact; retrain using this project")
    metrics = bundle["metrics"]
    score = float(bundle["model"].predict(pd.DataFrame({"Hours": [hours]}))[0])
    outside = not metrics["training_hours_min"] <= hours <= metrics["training_hours_max"]
    warnings = []
    if outside:
        warnings.append("Outside the training study-hour range; this is extrapolation.")
    if not 0 <= score <= 100:
        warnings.append("Linear prediction falls outside the 0-100 score scale; it is not clipped.")
    return {
        "hours": hours, "predicted_score": score,
        "extrapolation": outside, "warnings": warnings,
        "note": "Educational estimate from practice data, not a guaranteed exam result.",
    }
