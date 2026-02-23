"""
Unit Tests - MLflow GenAI Evaluation
====================================
"""

from src import genai_evaluation


class DummyCorrectness:
    pass


def test_default_eval_dataset_shape():
    data = genai_evaluation.default_eval_dataset()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "inputs" in data[0]
    assert "question" in data[0]["inputs"]
    assert "expectations" in data[0]
    assert "expected_response" in data[0]["expectations"]


def test_run_genai_evaluation_with_experiment_id(monkeypatch):
    calls = {}

    def fake_set_tracking_uri(uri):
        calls["tracking_uri"] = uri

    def fake_set_experiment(**kwargs):
        calls["set_experiment"] = kwargs

    def fake_evaluate(*, data, predict_fn, scorers):
        calls["evaluate_data"] = data
        calls["evaluate_predict_fn"] = predict_fn
        calls["evaluate_scorers"] = scorers
        return {"ok": True}

    monkeypatch.setattr(genai_evaluation.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(genai_evaluation.mlflow, "set_experiment", fake_set_experiment)
    monkeypatch.setattr(genai_evaluation.mlflow.genai, "evaluate", fake_evaluate)
    monkeypatch.setattr(genai_evaluation, "Correctness", DummyCorrectness)

    def predict_fn(question: str) -> str:
        return f"answer:{question}"

    result = genai_evaluation.run_genai_evaluation(
        predict_fn=predict_fn,
        tracking_uri="http://localhost:9878",
        experiment_id="1",
    )

    assert result == {"ok": True}
    assert calls["tracking_uri"] == "http://localhost:9878"
    assert calls["set_experiment"] == {"experiment_id": "1"}
    assert calls["evaluate_predict_fn"] is predict_fn
    assert len(calls["evaluate_scorers"]) == 2
    assert isinstance(calls["evaluate_scorers"][0], DummyCorrectness)


def test_run_genai_evaluation_with_experiment_name(monkeypatch):
    calls = {}

    def fake_set_experiment(**kwargs):
        calls["set_experiment"] = kwargs

    monkeypatch.setattr(genai_evaluation.mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(genai_evaluation.mlflow, "set_experiment", fake_set_experiment)
    monkeypatch.setattr(
        genai_evaluation.mlflow.genai,
        "evaluate",
        lambda **_: {"ok": True},
    )

    genai_evaluation.run_genai_evaluation(
        predict_fn=lambda question: question,
        experiment_id=None,
        experiment_name="imdb-recommender",
        scorers=[],
    )

    assert calls["set_experiment"] == {"experiment_name": "imdb-recommender"}


def test_expected_response_match_scorer():
    score_ok = genai_evaluation.expected_response_match(
        outputs="You can log a model by using the mlflow.sklearn.log_model function.",
        expectations={
            "expected_response": "You can log a model by using the mlflow.sklearn.log_model function."
        },
    )
    score_ko = genai_evaluation.expected_response_match(
        outputs="I do not know.",
        expectations={
            "expected_response": "You can log a model by using the mlflow.sklearn.log_model function."
        },
    )
    assert score_ok == 1.0
    assert score_ko == 0.0
