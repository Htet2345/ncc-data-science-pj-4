import json

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageChops
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from score_prediction.model import DEFAULT_DATA, predict, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("trained")
    metrics = train(output_dir=output)
    return output, metrics


def test_disjoint_reproducible_split_and_training_only_fit(trained):
    output, metrics = trained
    training = metrics["train_row_indices"]
    testing = metrics["test_row_indices"]
    assert len(training) == 25 and len(testing) == 7
    assert not set(training) & set(testing)
    assert set(training) | set(testing) == set(range(32))
    frame = pd.read_csv(DEFAULT_DATA)
    reference = LinearRegression().fit(frame.loc[training, ["Hours"]], frame.loc[training, "Scores"])
    model = joblib.load(output / "model.joblib")["model"]
    assert model.coef_[0] == pytest.approx(reference.coef_[0])
    assert model.intercept_ == pytest.approx(reference.intercept_)


def test_metrics_match_held_out_predictions_and_baseline(trained):
    output, metrics = trained
    results = pd.read_csv(output / "test_predictions.csv")
    assert metrics["mae"] == pytest.approx(mean_absolute_error(results.Actual, results.Predicted))
    assert metrics["rmse"] == pytest.approx(root_mean_squared_error(results.Actual, results.Predicted))
    assert list(results.row_index) == metrics["test_row_indices"]
    assert np.allclose(results.Residual, results.Actual - results.Predicted)
    frame = pd.read_csv(DEFAULT_DATA)
    mean_train_score = frame.loc[metrics["train_row_indices"], "Scores"].mean()
    assert metrics["baseline_mae"] == pytest.approx(mean_absolute_error(results.Actual, [mean_train_score] * len(results)))
    # Applies to the deliberately simple bundled synthetic dataset only.
    assert metrics["mae"] < metrics["baseline_mae"]
    assert json.loads((output / "metrics.json").read_text()) == metrics


def test_saved_prediction_matches_fitted_equation(trained):
    output, metrics = trained
    result = predict(9.25, output / "model.joblib")
    assert result["predicted_score"] == pytest.approx(metrics["intercept"] + metrics["coefficient"] * 9.25)
    assert result["warnings"] == []


@pytest.mark.parametrize("hours", [-1, 25, float("nan"), float("inf"), -float("inf")])
def test_invalid_prediction_hours(trained, hours):
    with pytest.raises(ValueError, match="finite number"):
        predict(hours, trained[0] / "model.joblib")


@pytest.mark.parametrize("hours", [0, 24])
def test_extrapolation_is_explicit_and_not_silently_clipped(trained, hours):
    result = predict(hours, trained[0] / "model.joblib")
    assert result["extrapolation"]
    assert result["warnings"]
    if hours == 24:
        assert result["predicted_score"] > 100
        assert len(result["warnings"]) == 2


def test_charts_are_decodable_nonblank_images(trained):
    for filename in ["regression.png", "residuals.png"]:
        with Image.open(trained[0] / filename) as image:
            assert image.format == "PNG" and image.size == (1600, 960)
            assert ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox()


def test_known_linear_relationship_and_test_targets_do_not_affect_fit(tmp_path):
    frame = pd.DataFrame({"Hours": np.linspace(1, 10, 30)})
    frame["Scores"] = 5 + 7 * frame.Hours
    path = tmp_path / "known.csv"
    frame.to_csv(path, index=False)
    first = train(path, tmp_path / "first")
    assert first["coefficient"] == pytest.approx(7)
    assert first["intercept"] == pytest.approx(5)
    assert first["mae"] == pytest.approx(0, abs=1e-10)
    assert predict(4.2, tmp_path / "first" / "model.joblib")["predicted_score"] == pytest.approx(34.4)
    frame.loc[first["test_row_indices"], "Scores"] = 20
    frame.to_csv(path, index=False)
    second = train(path, tmp_path / "second")
    assert second["train_row_indices"] == first["train_row_indices"]
    assert second["coefficient"] == pytest.approx(first["coefficient"])
    assert second["intercept"] == pytest.approx(first["intercept"])
    assert second["mae"] > first["mae"]
    assert second["r2"] is None


def test_missing_model_is_not_automatically_retrained(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict(5, tmp_path / "missing.joblib")
