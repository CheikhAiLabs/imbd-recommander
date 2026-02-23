"""
MLflow GenAI Evaluation Utilities
=================================
Helpers to run scorer-based quality evaluation (e.g., Correctness) on
question/answer style outputs and log results to MLflow.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import mlflow
from mlflow.genai.scorers import Correctness, scorer

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:9878")
DEFAULT_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "1")
DEFAULT_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "imdb-recommender")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@scorer(
    name="expected_response_match",
    description="Checks whether the expected response appears in model output.",
    aggregations=["mean"],
)
def expected_response_match(outputs: Any, expectations: dict[str, Any] | None = None) -> float:
    """Custom scorer that produces a deterministic quality metric."""
    expected = ((expectations or {}).get("expected_response") or "").strip().lower()
    rendered_output = (str(outputs) if outputs is not None else "").strip().lower()
    if not expected:
        return 0.0
    return 1.0 if expected in rendered_output else 0.0


def default_scorers() -> list[Any]:
    """Return default scorer set for evaluation."""
    return [Correctness(), expected_response_match]


def default_eval_dataset() -> list[dict[str, Any]]:
    """Return a minimal dataset for scorer-based Q/A evaluation."""
    return [
        {
            "inputs": {"question": "How do I log a model with MLflow?"},
            "expectations": {
                "expected_response": (
                    "You can log a model by using the mlflow.<flavor>.log_model function."
                )
            },
        }
    ]


def openai_predict(question: str, model: str = DEFAULT_OPENAI_MODEL) -> str:
    """
    Predict function compatible with ``mlflow.genai.evaluate``.

    Uses the OpenAI Chat Completions API and returns assistant text content.
    """
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for OpenAI-based evaluation. "
            "Install dependencies from requirements.txt."
        ) from exc

    response = openai.OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or ""


def run_genai_evaluation(
    predict_fn: Callable[[str], str],
    *,
    tracking_uri: str | None = None,
    experiment_id: str | None = DEFAULT_EXPERIMENT_ID,
    experiment_name: str | None = None,
    data: list[dict[str, Any]] | None = None,
    scorers: list[Any] | None = None,
) -> Any:
    """
    Run MLflow GenAI evaluation and log scorer metrics.

    Args:
        predict_fn: Function receiving ``question: str`` and returning text answer.
        tracking_uri: MLflow tracking URI.
        experiment_id: Target experiment ID (preferred for stable routing).
        experiment_name: Fallback experiment name when ``experiment_id`` is omitted.
        data: Evaluation dataset in MLflow GenAI format.
        scorers: Optional scorers list. Defaults to built-in ``Correctness()``
            plus custom ``expected_response_match``.
    """
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)

    if experiment_id:
        mlflow.set_experiment(experiment_id=str(experiment_id))
    else:
        mlflow.set_experiment(experiment_name=experiment_name or DEFAULT_EXPERIMENT_NAME)

    return mlflow.genai.evaluate(
        data=data or default_eval_dataset(),
        predict_fn=predict_fn,
        scorers=scorers or default_scorers(),
    )
