"""
Feature Engineering
====================
Preprocesses raw IMDb data into model-ready features.
Handles filtering, encoding, scaling, and text feature creation.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

logger = logging.getLogger(__name__)


def filter_titles(
    df: pd.DataFrame,
    title_types: list[str] | None = None,
    min_votes: int = 100,
    min_year: int = 1970,
    exclude_adult: bool = True,
) -> pd.DataFrame:
    """
    Filter titles to keep only relevant content.

    Args:
        df: Raw merged DataFrame.
        title_types: List of title types to keep (e.g., ['movie', 'tvSeries']).
        min_votes: Minimum number of votes.
        min_year: Earliest start year to include.
        exclude_adult: Whether to exclude adult content.

    Returns:
        Filtered DataFrame.
    """
    original_count = len(df)

    if title_types is None:
        title_types = ["movie", "tvSeries", "tvMiniSeries", "tvMovie"]

    mask = (
        df["titleType"].isin(title_types)
        & (df["numVotes"] >= min_votes)
        & (df["startYear"] >= min_year)
        & df["startYear"].notna()
        & df["genres"].notna()
        & df["runtimeMinutes"].notna()
        & (df["runtimeMinutes"] > 0)
    )

    if exclude_adult:
        mask = mask & (df["isAdult"] == 0)

    df_filtered = df[mask].copy().reset_index(drop=True)
    logger.info(
        f"Filtered titles: {original_count:,} -> {len(df_filtered):,} "
        f"({len(df_filtered) / original_count * 100:.1f}%)"
    )
    return df_filtered


def encode_genres(df: pd.DataFrame) -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    """
    One-hot encode the genres column (comma-separated string).

    Returns:
        Tuple of (DataFrame with genre columns, fitted MLB encoder).
    """
    logger.info("Encoding genres...")

    # Split genres string into lists
    genre_lists = df["genres"].str.split(",")

    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(genre_lists),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index,
    )

    df_with_genres = pd.concat([df, genre_encoded], axis=1)
    logger.info(f"Encoded {len(mlb.classes_)} genres: {list(mlb.classes_)}")
    return df_with_genres, mlb


def scale_numerical_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, MinMaxScaler]]:
    """
    Scale numerical features to [0, 1] range using MinMaxScaler.

    Returns:
        Tuple of (DataFrame with scaled features, dict of fitted scalers).
    """
    if columns is None:
        columns = ["averageRating", "numVotes", "runtimeMinutes", "startYear"]

    scalers = {}
    df = df.copy()

    for col in columns:
        if col in df.columns:
            scaler = MinMaxScaler()
            df[f"{col}_scaled"] = scaler.fit_transform(df[[col]].fillna(df[col].median()))
            scalers[col] = scaler
            logger.info(f"Scaled {col}: range [{df[col].min():.1f}, {df[col].max():.1f}] -> [0, 1]")

    return df, scalers


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined text feature for semantic embedding.
    Combines title, genres, year, and other metadata into a single string.
    """
    df = df.copy()

    df["text_feature"] = (
        df["primaryTitle"].fillna("")
        + " | "
        + df["genres"].fillna("")
        + " | Year: "
        + df["startYear"].fillna(0).astype(int).astype(str)
        + " | Rating: "
        + df["averageRating"].fillna(0).astype(str)
    )

    logger.info("Created text features for embedding")
    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the final feature matrix for similarity computation.

    Uses scaled numerical features and one-hot encoded genres.

    Returns:
        Tuple of (feature matrix as numpy array, list of feature names).
    """
    # Identify all feature columns
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    scaled_cols = [c for c in df.columns if c.endswith("_scaled")]

    feature_cols = scaled_cols + genre_cols

    if not feature_cols:
        raise ValueError("No feature columns found. Run encoding and scaling first.")

    feature_matrix = df[feature_cols].fillna(0).values.astype(np.float32)
    logger.info(
        f"Feature matrix shape: {feature_matrix.shape} "
        f"({len(scaled_cols)} numerical + {len(genre_cols)} genre features)"
    )

    return feature_matrix, feature_cols


def run_feature_pipeline(
    df: pd.DataFrame,
    min_votes: int = 100,
    min_year: int = 1970,
    title_types: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw merged DataFrame.
        min_votes: Minimum votes filter.
        min_year: Minimum year filter.
        title_types: Title types to include.

    Returns:
        Tuple of (processed DataFrame, feature matrix, feature names, artifacts dict).
    """
    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 60)

    # Step 1: Filter
    df = filter_titles(df, title_types=title_types, min_votes=min_votes, min_year=min_year)

    # Step 2: Encode genres
    df, mlb = encode_genres(df)

    # Step 3: Scale numerical features
    df, scalers = scale_numerical_features(df)

    # Step 4: Create text features
    df = create_text_features(df)

    # Step 5: Build feature matrix
    feature_matrix, feature_names = build_feature_matrix(df)

    artifacts = {
        "mlb": mlb,
        "scalers": scalers,
        "feature_names": feature_names,
    }

    logger.info("Feature engineering pipeline complete")
    return df, feature_matrix, feature_names, artifacts
