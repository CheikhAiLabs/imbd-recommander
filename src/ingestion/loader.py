"""
Data Loader
============
Efficient loading and initial cleaning of raw IMDb TSV files.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# IMDb uses '\\N' as the null marker
IMDB_NULL = "\\N"


def load_title_basics(filepath: Path) -> pd.DataFrame:
    """
    Load title.basics.tsv with proper dtypes and null handling.

    Columns:
        tconst, titleType, primaryTitle, originalTitle,
        isAdult, startYear, endYear, runtimeMinutes, genres
    """
    logger.info(f"Loading title.basics from {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype={
            "tconst": str,
            "titleType": str,
            "primaryTitle": str,
            "originalTitle": str,
            "isAdult": str,
            "startYear": str,
            "endYear": str,
            "runtimeMinutes": str,
            "genres": str,
        },
        na_values=[IMDB_NULL],
        low_memory=False,
    )

    # Type conversions (after null handling)
    df["isAdult"] = pd.to_numeric(df["isAdult"], errors="coerce").fillna(0).astype(int)
    df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
    df["endYear"] = pd.to_numeric(df["endYear"], errors="coerce")
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")

    logger.info(f"Loaded title.basics: {len(df):,} rows, {df.shape[1]} columns")
    return df


def load_title_ratings(filepath: Path) -> pd.DataFrame:
    """
    Load title.ratings.tsv with proper dtypes.

    Columns:
        tconst, averageRating, numVotes
    """
    logger.info(f"Loading title.ratings from {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype={
            "tconst": str,
            "averageRating": float,
            "numVotes": int,
        },
        na_values=[IMDB_NULL],
    )

    logger.info(f"Loaded title.ratings: {len(df):,} rows")
    return df


def load_title_principals(filepath: Path) -> pd.DataFrame:
    """
    Load title.principals.tsv with proper dtypes.

    Columns:
        tconst, ordering, nconst, category, job, characters
    """
    logger.info(f"Loading title.principals from {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        na_values=[IMDB_NULL],
        low_memory=False,
    )

    logger.info(f"Loaded title.principals: {len(df):,} rows")
    return df


def load_and_merge_datasets(
    raw_dir: Path,
    include_principals: bool = False,
) -> pd.DataFrame:
    """
    Load and merge all required datasets into a single DataFrame.

    Args:
        raw_dir: Directory containing raw TSV files.
        include_principals: Whether to include principals data.

    Returns:
        Merged DataFrame with titles and ratings.
    """
    basics_path = raw_dir / "title.basics.tsv"
    ratings_path = raw_dir / "title.ratings.tsv"

    if not basics_path.exists():
        raise FileNotFoundError(f"title.basics.tsv not found in {raw_dir}")
    if not ratings_path.exists():
        raise FileNotFoundError(f"title.ratings.tsv not found in {raw_dir}")

    # Load datasets
    basics = load_title_basics(basics_path)
    ratings = load_title_ratings(ratings_path)

    # Merge on tconst
    df = basics.merge(ratings, on="tconst", how="inner")
    logger.info(f"Merged basics + ratings: {len(df):,} rows")

    if include_principals:
        principals_path = raw_dir / "title.principals.tsv"
        if principals_path.exists():
            principals = load_title_principals(principals_path)
            # Aggregate top cast/crew per title
            top_people = (
                principals.groupby("tconst")["nconst"]
                .apply(lambda x: ",".join(x.head(5)))
                .reset_index()
                .rename(columns={"nconst": "topCast"})
            )
            df = df.merge(top_people, on="tconst", how="left")
            logger.info(f"Merged with principals: {len(df):,} rows")

    return df
