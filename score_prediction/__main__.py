"""Train: python -m score_prediction train; predict: ... predict --hours 9.25"""

import argparse
import json
from pathlib import Path

from score_prediction.model import DEFAULT_DATA, DEFAULT_OUTPUT, predict, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict student scores from study hours.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and evaluate a regression model")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA)
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prediction = commands.add_parser("predict", help="Use a trusted local trained model")
    prediction.add_argument("--hours", type=float, required=True)
    prediction.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.joblib")
    args = parser.parse_args(argv)
    try:
        result = train(args.data, args.output_dir) if args.command == "train" else predict(args.hours, args.model)
    except FileNotFoundError as error:
        parser.error(f"File not found: {error.filename}. Run training first if the model is missing.")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
