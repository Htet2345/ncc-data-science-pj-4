import json
import subprocess
import sys

import pytest

from score_prediction.model import ROOT


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "score_prediction", *args], cwd=ROOT, capture_output=True, text=True, timeout=90)


def test_train_then_predict_in_separate_processes(tmp_path):
    training = run_cli("train", "--output-dir", str(tmp_path))
    assert training.returncode == 0, training.stderr
    metrics = json.loads(training.stdout)
    assert metrics["rows"] == 32
    result = run_cli("predict", "--hours", "9.25", "--model", str(tmp_path / "model.joblib"))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["predicted_score"] > 0


@pytest.mark.parametrize("hours", ["-1", "25", "nan"])
def test_invalid_prediction_cli_errors_cleanly(hours):
    result = run_cli("predict", "--hours", hours)
    assert result.returncode == 2
    assert "error:" in result.stderr and "Traceback" not in result.stderr


def test_missing_model_cli_error(tmp_path):
    result = run_cli("predict", "--hours", "5", "--model", str(tmp_path / "missing.joblib"))
    assert result.returncode == 2
    assert "Run training first" in result.stderr


def test_missing_data_cli_error(tmp_path):
    result = run_cli("train", "--data", str(tmp_path / "missing.csv"))
    assert result.returncode == 2
    assert "File not found" in result.stderr
