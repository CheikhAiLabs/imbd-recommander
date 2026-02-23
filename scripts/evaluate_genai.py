#!/usr/bin/env python3
"""
Run MLflow GenAI scorer evaluation for question/answer quality metrics.
"""

from __future__ import annotations

import argparse
import json

from src.genai_evaluation import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TRACKING_URI,
    openai_predict,
    run_genai_evaluation,
)


def _mock_predict(question: str) -> str:
    return (
        "You can log a model by using the mlflow.<flavor>.log_model function. "
        f"(question={question})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MLflow GenAI evaluation with scorers.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="MLflow tracking URI")
    parser.add_argument(
        "--experiment-id",
        default=DEFAULT_EXPERIMENT_ID,
        help="MLflow experiment id (e.g. 1)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mock"],
        default="openai",
        help="Predict backend used during evaluation",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name when provider=openai",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    predict_fn = (
        (lambda question: openai_predict(question, model=args.openai_model))
        if args.provider == "openai"
        else _mock_predict
    )

    result = run_genai_evaluation(
        predict_fn=predict_fn,
        tracking_uri=args.tracking_uri,
        experiment_id=args.experiment_id,
    )

    print("MLflow GenAI evaluation completed.")
    if hasattr(result, "metrics"):
        print(json.dumps(result.metrics, indent=2, default=str))
    else:
        print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
