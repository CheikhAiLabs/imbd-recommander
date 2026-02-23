"""
Training Pipeline
==================
End-to-end reproducible pipeline for building the recommendation model.
Orchestrates data loading, feature engineering, model fitting, and artifact storage.
Fully integrated with MLflow for experiment tracking and model versioning.
"""

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Matplotlib backend for headless environments
import matplotlib
import numpy as np
import pandas as pd
import yaml

from src.features.engineering import run_feature_pipeline
from src.ingestion.downloader import download_all_datasets
from src.ingestion.loader import load_and_merge_datasets
from src.mlflow_tracking import (
    compute_data_statistics,
    compute_feature_analysis,
    compute_similarity_stats,
    init_mlflow,
    log_training_run,
)
from src.models.recommender import ContentRecommender
from src.validation.schemas import (
    FeatureEngineeredSchema,
    MergedDatasetSchema,
    validate_dataframe,
)

matplotlib.use("Agg")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/processed/training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using defaults")
        return get_default_config()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def get_default_config() -> dict:
    """Return default training configuration."""
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "include_principals": False,
            "force_download": False,
        },
        "features": {
            "min_votes": 500,
            "min_year": 1970,
            "title_types": ["movie", "tvSeries", "tvMiniSeries"],
        },
        "model": {
            "name": "content_recommender_v1",
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555"),
            "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "imdb-recommender"),
        },
    }


def run_pipeline(config: dict | None = None) -> dict:
    """
    Execute the full training pipeline.

    Steps:
        1. Download data (idempotent)
        2. Load and merge datasets
        3. Feature engineering
        4. Model fitting
        5. Save artifacts
        6. Validation

    Args:
        config: Training configuration dict.

    Returns:
        Summary dict with pipeline results.
    """
    start_time = time.time()
    config = config or get_default_config()

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("IMDb RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # ─── MLflow Init ──────────────────────────────────────────────────────
    mlflow_enabled = config.get("mlflow", {}).get("enabled", True)
    if mlflow_enabled:
        try:
            init_mlflow(
                tracking_uri=config["mlflow"].get("tracking_uri"),
                experiment_name=config["mlflow"].get("experiment_name"),
            )
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"MLflow init failed (continuing without tracking): {e}")
            mlflow_enabled = False

    # ─── Step 1: Data Download ────────────────────────────────────────────
    logger.info("\n📥 Step 1/6: Downloading datasets...")
    download_all_datasets(
        raw_dir=raw_dir,
        force=config["data"].get("force_download", False),
        include_principals=config["data"].get("include_principals", False),
    )

    # ─── Step 2: Data Loading ─────────────────────────────────────────────
    logger.info("\n📂 Step 2/6: Loading and merging datasets...")
    df_raw = load_and_merge_datasets(
        raw_dir=raw_dir,
        include_principals=config["data"].get("include_principals", False),
    )
    logger.info(f"Raw data: {len(df_raw):,} titles")

    # ─── Step 2b: Data Validation (Pandera) ───────────────────────────────
    logger.info("\n🛡️  Validating merged dataset (Pandera)...")
    try:
        validate_dataframe(df_raw, MergedDatasetSchema)
    except Exception as e:
        logger.warning(f"Merged data validation warning: {e}")

    # ─── Step 3: Feature Engineering ──────────────────────────────────────
    logger.info("\n⚙️  Step 3/6: Feature engineering...")
    df_processed, feature_matrix, feature_names, artifacts = run_feature_pipeline(
        df=df_raw,
        min_votes=config["features"]["min_votes"],
        min_year=config["features"]["min_year"],
        title_types=config["features"].get("title_types"),
    )

    # ─── Step 3b: Feature Validation (Pandera) ───────────────────────────
    logger.info("\n🛡️  Validating engineered dataset (Pandera)...")
    try:
        validate_dataframe(df_processed, FeatureEngineeredSchema)
    except Exception as e:
        logger.warning(f"Feature data validation warning: {e}")

    # ─── Step 4: Model Fitting ────────────────────────────────────────────
    logger.info("\n🔧 Step 4/6: Fitting recommendation model...")
    model = ContentRecommender()
    model.fit(
        feature_matrix=feature_matrix,
        tconst_ids=df_processed["tconst"].tolist(),
        titles=df_processed["primaryTitle"].tolist(),
        metadata={
            "feature_names": feature_names,
            "n_titles": len(df_processed),
            "trained_at": datetime.now(UTC).isoformat(),
            "config": config,
        },
    )

    # ─── Step 5: Save Artifacts ───────────────────────────────────────────
    logger.info("\n💾 Step 5/6: Saving artifacts...")

    # Save model
    model_path = processed_dir / "recommender_model.pkl"
    model.save(model_path)

    # Save title metadata (for enriched API responses)
    metadata_cols = [
        "tconst",
        "primaryTitle",
        "titleType",
        "startYear",
        "runtimeMinutes",
        "genres",
        "averageRating",
        "numVotes",
    ]
    available_cols = [c for c in metadata_cols if c in df_processed.columns]
    metadata_df = df_processed[available_cols].copy()
    metadata_path = processed_dir / "title_metadata.parquet"
    metadata_df.to_parquet(metadata_path, index=False)
    logger.info(f"Saved metadata: {metadata_path}")

    # Save pipeline manifest
    manifest = {
        "pipeline_version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "n_titles": len(df_processed),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "config": config,
    }
    manifest_path = processed_dir / "pipeline_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {manifest_path}")

    # ─── Step 6: Validation ───────────────────────────────────────────────
    logger.info("\n✅ Step 6/6: Validating model...")
    validation_results = _validate_model(model, df_processed)

    # ─── Step 7: MLflow Logging ───────────────────────────────────────────
    elapsed = time.time() - start_time
    mlflow_run_id = None
    if mlflow_enabled:
        logger.info("\n📊 Step 7/7: Logging to MLflow (maximum observability)...")
        try:
            # ── Core metrics ──
            metrics = {
                "n_titles": len(df_processed),
                "n_features": len(feature_names),
                "n_raw_titles": len(df_raw),
                "filter_ratio_pct": round(len(df_processed) / len(df_raw) * 100, 2),
                "pipeline_elapsed_seconds": round(elapsed, 2),
                "avg_rating": round(float(df_processed["averageRating"].mean()), 4),
                "std_rating": round(float(df_processed["averageRating"].std()), 4),
                "median_votes": float(df_processed["numVotes"].median()),
                "mean_votes": float(df_processed["numVotes"].mean()),
                "max_votes": float(df_processed["numVotes"].max()),
                "n_genres": len([c for c in df_processed.columns if c.startswith("genre_")]),
                "feature_matrix_size_mb": round(feature_matrix.nbytes / 1e6, 2),
                "feature_matrix_density_pct": round(
                    float(np.count_nonzero(feature_matrix) / feature_matrix.size * 100), 2
                ),
                "model_file_size_mb": round(model_path.stat().st_size / 1e6, 2),
                "metadata_file_size_mb": round(metadata_path.stat().st_size / 1e6, 2),
                "validation_passed": 1.0 if validation_results["passed"] else 0.0,
            }

            # Add year range info
            if "startYear" in df_processed.columns:
                years = df_processed["startYear"].dropna()
                metrics["year_min"] = int(years.min())
                metrics["year_max"] = int(years.max())
                metrics["year_mean"] = round(float(years.mean()), 1)

            # Add runtime info
            if "runtimeMinutes" in df_processed.columns:
                runtime = df_processed["runtimeMinutes"].dropna()
                if len(runtime) > 0:
                    metrics["runtime_mean"] = round(float(runtime.mean()), 1)
                    metrics["runtime_median"] = round(float(runtime.median()), 1)

            # Log validation check details
            for check in validation_results.get("checks", []):
                name = check.get("name", "unknown")
                metrics[f"validation.{name}"] = 1.0 if check.get("passed") else 0.0

            # ── Compute comprehensive data statistics ──
            logger.info("Computing data distribution statistics...")
            data_stats = compute_data_statistics(df_processed)

            # ── Compute feature analysis ──
            logger.info("Computing feature analysis...")
            feature_analysis = compute_feature_analysis(feature_matrix, feature_names)

            # ── Compute similarity distribution ──
            logger.info("Computing similarity score distribution (sample)...")
            similarity_stats = compute_similarity_stats(model)

            # Merge similarity stats into metrics
            for key, value in similarity_stats.get("similarity", {}).items():
                metrics[f"similarity.{key}"] = value

            mlflow_run_id = log_training_run(
                config=config,
                metrics=metrics,
                artifacts_dir=str(processed_dir),
                df_processed=df_processed,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
                model=model,
                data_stats=data_stats,
                feature_importance=feature_analysis,
                model_metadata={
                    "n_titles": len(df_processed),
                    "n_features": len(feature_names),
                    "model_name": config["model"]["name"],
                    "feature_matrix_shape": str(feature_matrix.shape),
                },
                tags={
                    "model.name": config["model"]["name"],
                    "data.min_votes": str(config["features"]["min_votes"]),
                    "data.min_year": str(config["features"]["min_year"]),
                    "data.source": "IMDb Non-Commercial Datasets",
                    "data.n_raw": str(len(df_raw)),
                    "data.n_processed": str(len(df_processed)),
                    "pipeline.elapsed": f"{elapsed:.1f}s",
                },
            )
            logger.info(f"MLflow run logged: {mlflow_run_id}")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Titles indexed: {len(df_processed):,}")
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Artifacts saved to: {processed_dir}")
    logger.info("=" * 70)

    return {
        "manifest": manifest,
        "validation": validation_results,
        "mlflow_run_id": mlflow_run_id,
    }


def _validate_model(
    model: ContentRecommender,
    df: pd.DataFrame,
) -> dict:
    """
    Run basic validation checks on the trained model.
    """
    results = {"passed": True, "checks": []}

    # Check 1: Model can recommend
    try:
        # Find a well-known movie to test
        test_titles = ["Inception", "The Matrix", "Breaking Bad", "The Dark Knight"]
        test_title = None
        for t in test_titles:
            if model.find_title_index(t) is not None:
                test_title = t
                break

        if test_title:
            recs = model.recommend(test_title, top_k=5)
            check = {
                "name": "recommendation_test",
                "query": test_title,
                "n_results": len(recs),
                "passed": len(recs) == 5,
                "top_rec": recs[0]["title"] if recs else None,
            }
            results["checks"].append(check)
            logger.info(
                f"Validation - '{test_title}' -> top rec: '{recs[0]['title']}' "
                f"(score: {recs[0]['similarity_score']:.4f})"
            )
        else:
            results["checks"].append(
                {"name": "recommendation_test", "passed": False, "reason": "No test title found"}
            )
            results["passed"] = False
    except Exception as e:
        results["checks"].append({"name": "recommendation_test", "passed": False, "error": str(e)})
        results["passed"] = False

    # Check 2: No NaN in feature matrix
    has_nan = np.isnan(model.feature_matrix).any()
    results["checks"].append({"name": "no_nan_features", "passed": not has_nan})
    if has_nan:
        results["passed"] = False
        logger.warning("Feature matrix contains NaN values!")

    # Check 3: Reasonable number of titles
    n_titles = len(model.titles)
    reasonable = n_titles > 1000
    results["checks"].append(
        {"name": "sufficient_titles", "n_titles": n_titles, "passed": reasonable}
    )

    return results


def main():
    """CLI entry point for training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="IMDb Recommender Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of datasets",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.force_download:
        config["data"]["force_download"] = True

    results = run_pipeline(config)

    if results["validation"]["passed"]:
        logger.info("All validation checks PASSED ✅")
    else:
        logger.warning("Some validation checks FAILED ⚠️")
        sys.exit(1)


if __name__ == "__main__":
    main()
