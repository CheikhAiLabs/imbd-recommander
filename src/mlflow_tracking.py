"""
MLflow v3 Integration — Maximum Observability
===============================================
Comprehensive MLflow configuration leveraging v3.10+ features:
  - Experiment tracking with 60+ metrics per training run
  - Dataset logging via ``mlflow.data.from_pandas``
  - Model registry via ``mlflow.pyfunc.log_model`` with signature
  - Rich artifacts: charts (``log_figure``), tables (``log_table``), JSON, logs
  - System metrics (CPU, memory) during training
  - Inference tracing via ``@mlflow.trace`` and ``mlflow.start_span``
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:9878")
DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "imdb-recommender")
REGISTERED_MODEL_NAME = "imdb-content-recommender"


# =============================================================================
#  Initialization
# =============================================================================


def init_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    enable_system_metrics: bool = True,
) -> str:
    """
    Initialize MLflow tracking with full v3.10 features.

    - Sets tracking URI and experiment
    - Enables system metrics logging (CPU, memory, disk, GPU)
    - Enables tracing
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    experiment = experiment_name or DEFAULT_EXPERIMENT

    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI: {uri}")

    # Enable system metrics (CPU, RAM, disk I/O) during runs
    if enable_system_metrics:
        try:
            mlflow.enable_system_metrics_logging()
            logger.info("MLflow system metrics logging enabled")
        except Exception as e:
            logger.warning(f"System metrics not available: {e}")

    # Create or get experiment
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        experiment_id = client.create_experiment(
            experiment,
            tags={
                "project": "imdb-recommender",
                "version": "2.0.0",
                "framework": "scikit-learn",
                "task": "content-based-recommendation",
            },
        )
        logger.info(f"Created MLflow experiment: {experiment} (id={experiment_id})")
    else:
        experiment_id = exp.experiment_id
        logger.info(f"Using MLflow experiment: {experiment} (id={experiment_id})")

    mlflow.set_experiment(experiment)
    return experiment_id


# =============================================================================
#  Training Run Logging  (called from train.py)
# =============================================================================


def log_training_run(
    config: dict,
    metrics: dict[str, float],
    artifacts_dir: str,
    df_processed: pd.DataFrame | None = None,
    feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    model: Any = None,
    data_stats: dict | None = None,
    feature_importance: dict | None = None,
    model_metadata: dict | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Log a complete training run to MLflow with **maximum** observability.

    V3.10 features used:
      - ``mlflow.log_input``        — log the training dataset
      - ``mlflow.log_table``        — log sample data & recommendations table
      - ``mlflow.log_figure``       — log matplotlib charts
      - ``mlflow.pyfunc.log_model`` — register model with signature
      - ``mlflow.log_text``         — log text summaries
      - System metrics              — CPU / RAM automatically tracked
    """
    try:
        run_context = mlflow.start_run(tags=tags or {})
    except Exception:
        # Fallback: disable system metrics if psutil is missing
        logger.warning("Falling back to start_run without system metrics")
        run_context = mlflow.start_run(tags=tags or {}, log_system_metrics=False)

    with run_context as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # ── 1) Parameters (flattened config) ──────────────────────────────
        flat_params = _flatten_dict(config, sep=".")
        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, value)
            except Exception:
                mlflow.log_param(key, str(value)[:250])

        # ── 2) Core metrics ───────────────────────────────────────────────
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and np.isnan(value)
            ):
                mlflow.log_metric(name, value)

        # ── 3) Dataset logging (mlflow.data) ─────────────────────────────
        if df_processed is not None:
            try:
                dataset = mlflow.data.from_pandas(
                    df_processed.head(5000),  # sample for size
                    source="IMDb Non-Commercial Datasets",
                    name="imdb-processed",
                    targets="averageRating",
                )
                mlflow.log_input(dataset, context="training")
                logger.info("Logged training dataset to MLflow")
            except Exception as e:
                logger.warning(f"Dataset logging failed: {e}")

        # ── 4) Data distribution stats as metrics ────────────────────────
        if data_stats:
            for key, value in data_stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"data.{key}", value)
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, (int, float)):
                            mlflow.log_metric(f"data.{key}.{sub_key}", sub_val)

            # JSON artifact
            stats_path = Path(artifacts_dir) / "data_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(_serialize_stats(data_stats), f, indent=2)
            mlflow.log_artifact(str(stats_path))

        # ── 5) Feature analysis as metrics + artifact ────────────────────
        if feature_importance:
            fi_path = Path(artifacts_dir) / "feature_analysis.json"
            with open(fi_path, "w") as f:
                json.dump(_serialize_stats(feature_importance), f, indent=2)
            mlflow.log_artifact(str(fi_path))

        # ── 6) Model metadata as params ──────────────────────────────────
        if model_metadata:
            for key, value in model_metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"model.{key}", value)

        # ── 7) Tables: sample data, genre distribution ───────────────────
        if df_processed is not None:
            _log_sample_tables(df_processed, model)

        # ── 8) Figures: charts ───────────────────────────────────────────
        if df_processed is not None:
            _log_distribution_figures(df_processed, feature_matrix, feature_names)

        # ── 9) Model registration with signature ────────────────────────
        if model is not None and df_processed is not None:
            _log_model_to_registry(model, df_processed, artifacts_dir)

        # ── 10) Text summaries ───────────────────────────────────────────
        _log_text_summary(metrics, config, data_stats)

        # ── 11) All file artifacts (.pkl, .parquet, .json, .log, .yaml) ──
        artifacts_path = Path(artifacts_dir)
        if artifacts_path.exists():
            for f in artifacts_path.iterdir():
                if f.is_file() and f.suffix in (".pkl", ".parquet", ".json", ".log", ".yaml"):
                    mlflow.log_artifact(str(f))
                    logger.info(f"Logged artifact: {f.name}")

        # ── 12) Summary tags ─────────────────────────────────────────────
        mlflow.set_tag("pipeline.status", "completed")
        mlflow.set_tag("pipeline.version", "2.0.0")
        mlflow.set_tag("mlflow.runName", f"train-{metrics.get('n_titles', '?')}-titles")
        mlflow.set_tag(
            "mlflow.note.content",
            f"Training completed: {metrics.get('n_titles', '?')} titles, "
            f"{metrics.get('n_features', '?')} features, "
            f"{metrics.get('pipeline_elapsed_seconds', '?')}s elapsed. "
            f"Model registered as '{REGISTERED_MODEL_NAME}'.",
        )

        logger.info(f"MLflow run completed: {run_id}")
        return run_id


# =============================================================================
#  Tables  (mlflow.log_table — new in v3)
# =============================================================================


def _log_sample_tables(df: pd.DataFrame, model: Any = None):
    """Log structured tables for exploration in MLflow UI."""
    try:
        # Table 1: Sample titles (top-rated)
        sample_cols = [
            "tconst",
            "primaryTitle",
            "titleType",
            "startYear",
            "averageRating",
            "numVotes",
            "genres",
        ]
        avail_cols = [c for c in sample_cols if c in df.columns]
        top_rated = df.nlargest(50, "averageRating")[avail_cols].reset_index(drop=True)
        mlflow.log_table(top_rated, artifact_file="tables/top_50_rated.json")
        logger.info("Logged table: top_50_rated")

        # Table 2: Most voted
        most_voted = df.nlargest(50, "numVotes")[avail_cols].reset_index(drop=True)
        mlflow.log_table(most_voted, artifact_file="tables/top_50_voted.json")
        logger.info("Logged table: top_50_voted")

        # Table 3: Genre distribution
        genre_cols = [c for c in df.columns if c.startswith("genre_")]
        if genre_cols:
            genre_counts = (
                pd.DataFrame(
                    {
                        "genre": [c.replace("genre_", "") for c in genre_cols],
                        "count": [int(df[c].sum()) for c in genre_cols],
                        "percentage": [round(float(df[c].mean() * 100), 2) for c in genre_cols],
                    }
                )
                .sort_values("count", ascending=False)
                .reset_index(drop=True)
            )
            mlflow.log_table(genre_counts, artifact_file="tables/genre_distribution.json")
            logger.info("Logged table: genre_distribution")

        # Table 4: Title type distribution
        if "titleType" in df.columns:
            type_dist = df["titleType"].value_counts().reset_index()
            type_dist.columns = ["title_type", "count"]
            type_dist["percentage"] = round(type_dist["count"] / len(df) * 100, 2)
            mlflow.log_table(type_dist, artifact_file="tables/title_type_distribution.json")

        # Table 5: Sample recommendations (if model available)
        if model is not None and hasattr(model, "recommend"):
            test_titles = ["Inception", "The Matrix", "Breaking Bad", "The Dark Knight", "Parasite"]
            recs_rows = []
            for t in test_titles:
                try:
                    recs = model.recommend(t, top_k=5)
                    for r in recs:
                        recs_rows.append(
                            {
                                "query": t,
                                "rank": r["rank"],
                                "recommended": r["title"],
                                "similarity": r["similarity_score"],
                                "tconst": r["tconst"],
                            }
                        )
                except Exception:
                    pass
            if recs_rows:
                recs_df = pd.DataFrame(recs_rows)
                mlflow.log_table(recs_df, artifact_file="tables/sample_recommendations.json")
                logger.info("Logged table: sample_recommendations")

        # Table 6: Decade breakdown
        if "startYear" in df.columns:
            years = df["startYear"].dropna()
            decades = (years // 10 * 10).astype(int)
            decade_dist = decades.value_counts().sort_index().reset_index()
            decade_dist.columns = ["decade", "count"]
            decade_dist["percentage"] = round(decade_dist["count"] / len(df) * 100, 2)
            mlflow.log_table(decade_dist, artifact_file="tables/decade_distribution.json")

    except Exception as e:
        logger.warning(f"Table logging failed: {e}")


# =============================================================================
#  Figures  (mlflow.log_figure — charts)
# =============================================================================


def _log_distribution_figures(
    df: pd.DataFrame,
    feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
):
    """Generate and log matplotlib charts to MLflow."""
    plt.style.use("dark_background")

    try:
        # Figure 1: Rating distribution
        if "averageRating" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(
                df["averageRating"].dropna(),
                bins=50,
                color="#f5c518",
                edgecolor="black",
                alpha=0.85,
            )
            ax.set_xlabel("Average Rating", fontsize=12)
            ax.set_ylabel("Number of Titles", fontsize=12)
            ax.set_title("IMDb Rating Distribution", fontsize=14, fontweight="bold")
            ax.axvline(
                df["averageRating"].mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {df['averageRating'].mean():.2f}",
            )
            ax.legend()
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/rating_distribution.png")
            plt.close(fig)
            logger.info("Logged figure: rating_distribution")

        # Figure 2: Vote distribution (log scale)
        if "numVotes" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(
                np.log10(df["numVotes"].dropna().clip(lower=1)),
                bins=50,
                color="#4fc3f7",
                edgecolor="black",
                alpha=0.85,
            )
            ax.set_xlabel("log\u2081\u2080(Number of Votes)", fontsize=12)
            ax.set_ylabel("Number of Titles", fontsize=12)
            ax.set_title("Vote Count Distribution (Log Scale)", fontsize=14, fontweight="bold")
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/votes_distribution.png")
            plt.close(fig)

        # Figure 3: Year/decade distribution
        if "startYear" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            years = df["startYear"].dropna()
            decades = (years // 10 * 10).astype(int)
            decade_counts = decades.value_counts().sort_index()
            ax.bar(
                decade_counts.index.astype(str),
                decade_counts.values,
                color="#66bb6a",
                edgecolor="black",
                alpha=0.85,
            )
            ax.set_xlabel("Decade", fontsize=12)
            ax.set_ylabel("Number of Titles", fontsize=12)
            ax.set_title("Titles by Decade", fontsize=14, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/decade_distribution.png")
            plt.close(fig)

        # Figure 4: Genre distribution (horizontal bar)
        genre_cols = [c for c in df.columns if c.startswith("genre_")]
        if genre_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            genre_names = [c.replace("genre_", "") for c in genre_cols]
            genre_counts = [int(df[c].sum()) for c in genre_cols]
            sorted_pairs = sorted(zip(genre_names, genre_counts, strict=False), key=lambda x: x[1])
            names, counts = zip(*sorted_pairs, strict=False)
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
            ax.barh(names, counts, color=colors, edgecolor="black", alpha=0.85)
            ax.set_xlabel("Number of Titles", fontsize=12)
            ax.set_title("Genre Distribution", fontsize=14, fontweight="bold")
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/genre_distribution.png")
            plt.close(fig)

        # Figure 5: Runtime distribution
        if "runtimeMinutes" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            runtime = df["runtimeMinutes"].dropna()
            runtime_clipped = runtime[runtime <= 300]  # Exclude outliers > 5h
            ax.hist(runtime_clipped, bins=60, color="#ef5350", edgecolor="black", alpha=0.85)
            ax.set_xlabel("Runtime (minutes)", fontsize=12)
            ax.set_ylabel("Number of Titles", fontsize=12)
            ax.set_title("Runtime Distribution", fontsize=14, fontweight="bold")
            ax.axvline(
                runtime.median(),
                color="yellow",
                linestyle="--",
                label=f"Median: {runtime.median():.0f} min",
            )
            ax.legend()
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/runtime_distribution.png")
            plt.close(fig)

        # Figure 6: Rating vs Votes scatter
        if "averageRating" in df.columns and "numVotes" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sample = df.sample(n=min(5000, len(df)), random_state=42)
            ax.scatter(
                np.log10(sample["numVotes"].clip(lower=1)),
                sample["averageRating"],
                alpha=0.3,
                s=10,
                c="#f5c518",
            )
            ax.set_xlabel("log\u2081\u2080(Votes)", fontsize=12)
            ax.set_ylabel("Rating", fontsize=12)
            ax.set_title("Rating vs Vote Count", fontsize=14, fontweight="bold")
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/rating_vs_votes.png")
            plt.close(fig)

        # Figure 7: Feature correlation heatmap
        if feature_matrix is not None and feature_names is not None:
            n_num = min(8, len(feature_names))
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = np.corrcoef(feature_matrix[:, :n_num].T)
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(n_num))
            ax.set_yticks(range(n_num))
            short_names = [n[:12] for n in feature_names[:n_num]]
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(short_names, fontsize=9)
            ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.8)
            for i in range(n_num):
                for j in range(n_num):
                    ax.text(
                        j,
                        i,
                        f"{corr[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(corr[i, j]) < 0.5 else "black",
                    )
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/feature_correlation.png")
            plt.close(fig)

        # Figure 8: Similarity score distribution (sampled)
        if feature_matrix is not None:
            fig = _plot_similarity_distribution(feature_matrix)
            if fig:
                mlflow.log_figure(fig, "figures/similarity_distribution.png")
                plt.close(fig)

    except Exception as e:
        logger.warning(f"Figure logging failed: {e}")
    finally:
        plt.close("all")


def _plot_similarity_distribution(feature_matrix: np.ndarray):
    """Plot the distribution of cosine similarities from a sample."""
    from sklearn.metrics.pairwise import cosine_similarity

    n = min(500, feature_matrix.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(feature_matrix.shape[0], size=n, replace=False)
    sample = feature_matrix[idx]
    sim = cosine_similarity(sample)
    mask = ~np.eye(n, dtype=bool)
    sim_values = sim[mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim_values, bins=80, color="#ab47bc", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Pairwise Similarity Distribution (sample of 500)", fontsize=14, fontweight="bold")
    ax.axvline(
        np.mean(sim_values),
        color="yellow",
        linestyle="--",
        label=f"Mean: {np.mean(sim_values):.4f}",
    )
    ax.axvline(
        np.median(sim_values),
        color="cyan",
        linestyle="--",
        label=f"Median: {np.median(sim_values):.4f}",
    )
    ax.legend()
    fig.tight_layout()
    return fig


# =============================================================================
#  Model Registry  (mlflow.pyfunc)
# =============================================================================


def _log_model_to_registry(model: Any, df: pd.DataFrame, artifacts_dir: str):
    """Log the model to MLflow Model Registry with signature."""
    try:
        # Build a signature: input = title string, output = list of recommendations
        input_example = pd.DataFrame({"title": ["Inception"], "top_k": [10]})
        output_example = pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "tconst": ["tt6723592", "tt0816692", "tt0468569"],
                "title": ["Tenet", "Interstellar", "The Dark Knight"],
                "similarity_score": [0.9847, 0.9712, 0.9456],
            }
        )
        signature = infer_signature(input_example, output_example)

        # Log model as a pyfunc with artifacts
        model_path = Path(artifacts_dir) / "recommender_model.pkl"
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="model")

        # Register with pyfunc flavor
        mlflow.pyfunc.log_model(
            artifact_path="recommender_pyfunc",
            python_model=_RecommenderPyfuncWrapper(model),
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            pip_requirements=[
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0",
                "pandas>=2.0.0",
            ],
        )
        logger.info(f"Model registered as '{REGISTERED_MODEL_NAME}' in MLflow Model Registry")
    except Exception as e:
        logger.warning(f"Model registry logging failed: {e}")


class _RecommenderPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper to make ContentRecommender compatible with MLflow pyfunc."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None):
        results = []
        for _, row in model_input.iterrows():
            title = row.get("title", "")
            top_k = int(row.get("top_k", 10))
            tconst = row.get("tconst", None)
            try:
                recs = self.model.recommend(title, top_k=top_k, tconst=tconst)
                for rec in recs:
                    rec["query"] = title
                results.extend(recs)
            except Exception as e:
                results.append({"query": title, "error": str(e)})
        return pd.DataFrame(results)


# =============================================================================
#  Text Summaries
# =============================================================================


def _log_text_summary(metrics: dict, config: dict, data_stats: dict | None):
    """Log a human-readable training summary."""
    try:
        lines = [
            "=" * 60,
            "  IMDb Recommender — Training Run Summary",
            "=" * 60,
            "",
            f"  Titles indexed:    {metrics.get('n_titles', '?'):>10,}",
            f"  Features:          {metrics.get('n_features', '?'):>10}",
            f"  Raw titles:        {metrics.get('n_raw_titles', '?'):>10,}",
            f"  Filter ratio:      {metrics.get('filter_ratio_pct', '?'):>9}%",
            f"  Elapsed time:      {metrics.get('pipeline_elapsed_seconds', '?'):>9}s",
            "",
            "  ── Data Distribution ──",
            f"  Avg rating:        {metrics.get('avg_rating', '?'):>10}",
            f"  Median votes:      {metrics.get('median_votes', '?'):>10,.0f}",
            f"  Year range:        {metrics.get('year_min', '?')} - {metrics.get('year_max', '?')}",
            "",
            "  ── Model Info ──",
            f"  Model size:        {metrics.get('model_file_size_mb', '?'):>9} MB",
            f"  Matrix density:    {metrics.get('feature_matrix_density_pct', '?'):>9}%",
            f"  Validation:        {'PASSED' if metrics.get('validation_passed') == 1.0 else 'FAILED'}",
            "",
            "  ── Similarity Stats ──",
            f"  Mean similarity:   {metrics.get('similarity.mean', '?')}",
            f"  Median similarity: {metrics.get('similarity.median', '?')}",
            f"  >50% similar:      {metrics.get('similarity.above_50pct', '?')}%",
            f"  >80% similar:      {metrics.get('similarity.above_80pct', '?')}%",
            "",
            "=" * 60,
        ]
        summary = "\n".join(lines)
        mlflow.log_text(summary, "training_summary.txt")
        logger.info("Logged training summary text")
    except Exception as e:
        logger.warning(f"Text summary logging failed: {e}")


# =============================================================================
#  Data & Feature Statistics  (reused from v1)
# =============================================================================


def compute_data_statistics(df: pd.DataFrame) -> dict:
    """Compute comprehensive data distribution statistics."""
    stats = {}

    if "averageRating" in df.columns:
        ratings = df["averageRating"].dropna()
        stats["rating"] = {
            "mean": round(float(ratings.mean()), 4),
            "median": round(float(ratings.median()), 4),
            "std": round(float(ratings.std()), 4),
            "min": round(float(ratings.min()), 4),
            "max": round(float(ratings.max()), 4),
            "q25": round(float(ratings.quantile(0.25)), 4),
            "q75": round(float(ratings.quantile(0.75)), 4),
            "skew": round(float(ratings.skew()), 4),
            "kurtosis": round(float(ratings.kurtosis()), 4),
        }

    if "numVotes" in df.columns:
        votes = df["numVotes"].dropna()
        stats["votes"] = {
            "mean": round(float(votes.mean()), 2),
            "median": round(float(votes.median()), 2),
            "std": round(float(votes.std()), 2),
            "min": int(votes.min()),
            "max": int(votes.max()),
            "q25": round(float(votes.quantile(0.25)), 2),
            "q75": round(float(votes.quantile(0.75)), 2),
            "q90": round(float(votes.quantile(0.90)), 2),
            "q99": round(float(votes.quantile(0.99)), 2),
            "log_mean": round(float(np.log1p(votes).mean()), 4),
            "total": int(votes.sum()),
        }

    if "startYear" in df.columns:
        years = df["startYear"].dropna()
        stats["year"] = {
            "min": int(years.min()),
            "max": int(years.max()),
            "mean": round(float(years.mean()), 1),
            "median": int(years.median()),
            "mode": int(years.mode().iloc[0]) if not years.mode().empty else 0,
            "n_decades": int((years.max() - years.min()) / 10) + 1,
        }
        decades = (years // 10 * 10).value_counts().sort_index()
        stats["decade_counts"] = {str(int(k)): int(v) for k, v in decades.items()}

    if "runtimeMinutes" in df.columns:
        runtime = df["runtimeMinutes"].dropna()
        if len(runtime) > 0:
            stats["runtime"] = {
                "mean": round(float(runtime.mean()), 1),
                "median": round(float(runtime.median()), 1),
                "std": round(float(runtime.std()), 1),
                "min": int(runtime.min()),
                "max": int(runtime.max()),
            }

    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    if genre_cols:
        genre_counts = {col.replace("genre_", ""): int(df[col].sum()) for col in genre_cols}
        stats["genre_counts"] = dict(sorted(genre_counts.items(), key=lambda x: -x[1]))
        stats["n_total_genres"] = len(genre_cols)
        stats["avg_genres_per_title"] = round(float(df[genre_cols].sum(axis=1).mean()), 2)

    if "titleType" in df.columns:
        type_counts = df["titleType"].value_counts()
        stats["title_type_counts"] = {str(k): int(v) for k, v in type_counts.items()}

    # Overall stats
    stats["n_titles"] = len(df)
    stats["n_columns"] = len(df.columns)
    stats["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 2)

    return stats


def compute_feature_analysis(feature_matrix: np.ndarray, feature_names: list[str]) -> dict:
    """Compute feature-level analysis."""
    analysis = {}

    feature_stats = {}
    for i, name in enumerate(feature_names):
        col = feature_matrix[:, i]
        feature_stats[name] = {
            "mean": round(float(np.mean(col)), 6),
            "std": round(float(np.std(col)), 6),
            "min": round(float(np.min(col)), 6),
            "max": round(float(np.max(col)), 6),
            "zeros_pct": round(float(np.sum(col == 0) / len(col) * 100), 2),
            "nonzero_pct": round(float(np.sum(col != 0) / len(col) * 100), 2),
        }
    analysis["feature_stats"] = feature_stats

    analysis["matrix"] = {
        "shape_rows": feature_matrix.shape[0],
        "shape_cols": feature_matrix.shape[1],
        "density_pct": round(
            float(np.count_nonzero(feature_matrix) / feature_matrix.size * 100), 2
        ),
        "memory_mb": round(feature_matrix.nbytes / 1e6, 2),
        "has_nan": bool(np.isnan(feature_matrix).any()),
        "has_inf": bool(np.isinf(feature_matrix).any()),
    }

    n_numerical = min(4, len(feature_names))
    if n_numerical > 1:
        corr = np.corrcoef(feature_matrix[:, :n_numerical].T)
        corr_dict = {}
        for i in range(n_numerical):
            for j in range(i + 1, n_numerical):
                key = f"{feature_names[i]}_vs_{feature_names[j]}"
                corr_dict[key] = round(float(corr[i, j]), 4)
        analysis["feature_correlations"] = corr_dict

    return analysis


def compute_similarity_stats(model) -> dict:
    """Compute cosine similarity distribution from a sample."""
    from sklearn.metrics.pairwise import cosine_similarity

    stats = {}
    n = min(500, len(model.titles))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(model.titles), size=n, replace=False)
    sample_matrix = model.feature_matrix[sample_idx]
    sim = cosine_similarity(sample_matrix)
    mask = ~np.eye(n, dtype=bool)
    sim_values = sim[mask]

    stats["similarity"] = {
        "mean": round(float(np.mean(sim_values)), 6),
        "median": round(float(np.median(sim_values)), 6),
        "std": round(float(np.std(sim_values)), 6),
        "min": round(float(np.min(sim_values)), 6),
        "max": round(float(np.max(sim_values)), 6),
        "q10": round(float(np.quantile(sim_values, 0.10)), 6),
        "q25": round(float(np.quantile(sim_values, 0.25)), 6),
        "q75": round(float(np.quantile(sim_values, 0.75)), 6),
        "q90": round(float(np.quantile(sim_values, 0.90)), 6),
        "q99": round(float(np.quantile(sim_values, 0.99)), 6),
        "above_50pct": round(float(np.mean(sim_values > 0.5) * 100), 2),
        "above_80pct": round(float(np.mean(sim_values > 0.8) * 100), 2),
        "sample_size": n,
    }

    return stats


# =============================================================================
#  Inference Tracing  (MLflow Tracing API — new in v3)
# =============================================================================


def init_inference_tracing(tracking_uri: str | None = None, experiment_name: str | None = None):
    """
    Initialize MLflow tracing for API inference.

    Enables the tracing backend so that ``@mlflow.trace`` decorators produce
    visible traces in the MLflow UI under the Traces tab.
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    experiment = experiment_name or DEFAULT_EXPERIMENT

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    # Enable tracing
    try:
        mlflow.tracing.enable()
        logger.info(f"MLflow inference tracing enabled -> {uri}")
    except Exception as e:
        logger.warning(f"MLflow tracing enable failed: {e}")


# =============================================================================
#  Helpers
# =============================================================================


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def _serialize_stats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serialize_stats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_stats(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
