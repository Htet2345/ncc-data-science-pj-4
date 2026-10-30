# Student Score Prediction - Data Science Project 4

Predict a student's score from daily study hours using pandas and scikit-learn
linear regression. This project covers the topic listed in Module 5.1, videos
6-7: **Predict Student Score from Study Hours, Parts 1-2**.

This is an original implementation based on that lesson title. The teacher's
exact code and dataset have not been verified. The bundled CSV contains 32
synthetic observations for practice, not records from real students.

## What the project does

1. Load and validate a `Hours,Scores` CSV with pandas.
2. Split it reproducibly into 80% training and 20% held-out test observations
   using `random_state=42` (25 training rows and 7 test rows for the sample CSV).
3. Fit `LinearRegression` on training rows only.
4. Evaluate held-out predictions with MAE, RMSE and R-squared.
5. Compare MAE with a baseline that always predicts the training mean score.
6. Save the model, metrics, held-out predictions and two charts.
7. Load that model to predict a score for a supplied study duration.

No web server or external account is required. This is a command-line machine
learning project, with a VS Code run/debug configuration.

## Setup on Windows

Requires Python 3.10+. Developed with Python 3.13.

Open this folder in VS Code, then choose **Terminal > New Terminal**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The environment is already installed on the original development machine. The
commands above recreate it on a fresh checkout. The lock file includes exact
tested runtime and test versions. `requirements.txt` lists direct runtime
dependencies; `requirements-dev.txt` adds pytest.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe -m score_prediction train
```

This prints evaluation results and creates:

| File under `artifacts/` | Contents |
| --- | --- |
| `model.joblib` | Training-fitted model and training-range metadata |
| `metrics.json` | Held-out metrics, baseline, equation and split indices |
| `test_predictions.csv` | Test row IDs, hours, actual/predicted scores and residuals |
| `regression.png` | Training/test observations and fitted regression line |
| `residuals.png` | Held-out actual-minus-predicted errors |

The model is not refitted on all data after evaluation. The saved artifact is
the same model whose performance was measured on the held-out test rows.

## Predict a score

Train first, then run:

```powershell
.\.venv\Scripts\python.exe -m score_prediction predict --hours 9.25
```

The output contains study hours, predicted score, an extrapolation flag and
any warnings. A prediction is a floating-point estimate, not a pass/fail result.
Input hours must be finite and between 0 and 24. Predictions outside the
training-hour range are explicitly marked as extrapolation. Linear regression
can return scores outside 0-100; these are flagged rather than silently clipped.

Default input/output paths are relative to the project folder. Run the commands
from this folder so Python can import the package. To use a different CSV:

```powershell
.\.venv\Scripts\python.exe -m score_prediction train --data .\sample_data\student_scores.csv --output-dir .\artifacts
.\.venv\Scripts\python.exe -m score_prediction predict --hours 5 --model .\artifacts\model.joblib
```

Only load model artifacts you created or trust: joblib uses pickle, which can
execute code when loading an untrusted file. Use the saved dependency versions
when reloading the model. Generated artifacts are excluded from Git and can be
recreated by training.

## Understand the results

The learned equation is:

```text
predicted score = intercept + coefficient * study hours
```

- **MAE**: average absolute prediction error, in score points. Lower is better.
- **RMSE**: error in score points that gives larger mistakes more weight.
- **R-squared**: performance relative to predicting the held-out target mean.
  It can be negative. If the test scores are constant, this project reports
  `null` because the value is not informative.
- **Baseline MAE**: error from always predicting the mean training score.

Good performance on the small, deliberately simple synthetic dataset does not
demonstrate real-world accuracy. Study hours alone do not account for prior
knowledge, teaching, sleep, exam difficulty and other influences. This exercise
does not establish that changing study time causes a particular score.

## Dataset validation

CSV columns must be exactly `Hours,Scores`, with at least 10 rows, at least five
distinct hour values and varying scores. Missing, nonnumeric and nonfinite
values, exact duplicate rows, hours outside 0-24 and scores outside 0-100 are
rejected before training. See `sample_data/README.md` for dataset provenance.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests cover invalid data, disjoint training/test rows, training-only fitting,
held-out metrics, baseline calculation, known linear equations, saved model
predictions, extrapolation, chart files and separate-process CLI execution.
One test changes only test targets and verifies that learned coefficients do
not change, detecting accidental training on held-out labels.

In VS Code, **F5** trains the model. **Terminal > Run Task** includes training,
prediction and testing tasks. No virtual-environment activation or PowerShell
execution-policy changes are needed.

## Structure

```text
score_prediction/
  data.py       CSV loading and validation
  model.py      Train/test split, model, metrics, persistence and prediction
  plots.py      Regression and residual figures
  __main__.py   Training/prediction command-line interface
sample_data/    Synthetic input CSV and its description
tests/          Data, model and command-line checks
.vscode/        Interpreter, run/debug and test settings
artifacts/      Generated model, metrics and charts (ignored by Git)
```

## GitHub submission

Include Python source, tests, the synthetic input dataset, README, requirements,
pytest and VS Code configuration. Do not include the virtual environment,
generated artifacts, caches or local test report. No demo screenshots are
included. Repository: https://github.com/Htet2345/ncc-data-science-pj-4

## References

- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [scikit-learn regression metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html)
