"""
Data Validation Schemas
========================
Pandera schemas for validating DataFrames at every stage of the pipeline.

Schemas enforce column presence, types, value ranges, and nullability
to catch data quality issues early — before they silently corrupt
the model or produce misleading metrics.

Stages validated:
    1. Raw ingestion  — title.basics, title.ratings
    2. Merged dataset — basics + ratings joined on tconst
    3. Post-filtering — after title type / votes / year filters
    4. Post-feature engineering — final model-ready DataFrame
"""

import logging

import pandera as pa
from pandera import Check, Column

logger = logging.getLogger(__name__)


# =============================================================================
#  1) Raw ingestion schemas
# =============================================================================


TitleBasicsSchema = pa.DataFrameSchema(
    columns={
        "tconst": Column(str, Check.str_startswith("tt"), nullable=False),
        "titleType": Column(str, nullable=False),
        "primaryTitle": Column(str, nullable=False),
        "originalTitle": Column(str, nullable=True),
        "isAdult": Column(int, Check.isin([0, 1]), nullable=False),
        "startYear": Column(float, Check.in_range(1800, 2100), nullable=True),
        "endYear": Column(float, Check.in_range(1800, 2100), nullable=True),
        "runtimeMinutes": Column(float, Check.greater_than(0), nullable=True),
        "genres": Column(str, nullable=True),
    },
    coerce=False,
    strict=False,  # allow extra columns
    name="TitleBasicsSchema",
)


TitleRatingsSchema = pa.DataFrameSchema(
    columns={
        "tconst": Column(str, Check.str_startswith("tt"), nullable=False),
        "averageRating": Column(float, Check.in_range(0.0, 10.0), nullable=False),
        "numVotes": Column(int, Check.greater_than(0), nullable=False),
    },
    coerce=False,
    strict=False,
    name="TitleRatingsSchema",
)


# =============================================================================
#  2) Merged dataset schema  (basics ⨝ ratings)
# =============================================================================

MergedDatasetSchema = pa.DataFrameSchema(
    columns={
        "tconst": Column(str, Check.str_startswith("tt"), nullable=False, unique=True),
        "titleType": Column(str, nullable=False),
        "primaryTitle": Column(str, nullable=False),
        "isAdult": Column(int, Check.isin([0, 1]), nullable=False),
        "startYear": Column(float, Check.in_range(1800, 2100), nullable=True),
        "runtimeMinutes": Column(float, Check.greater_than(0), nullable=True),
        "genres": Column(str, nullable=True),
        "averageRating": Column(float, Check.in_range(0.0, 10.0), nullable=False),
        "numVotes": Column(int, Check.greater_than(0), nullable=False),
    },
    coerce=False,
    strict=False,
    name="MergedDatasetSchema",
)


# =============================================================================
#  3) Post-filtering schema
# =============================================================================

FilteredDatasetSchema = pa.DataFrameSchema(
    columns={
        "tconst": Column(str, Check.str_startswith("tt"), nullable=False, unique=True),
        "titleType": Column(
            str,
            Check.isin(["movie", "tvSeries", "tvMiniSeries", "tvMovie", "tvSpecial"]),
            nullable=False,
        ),
        "primaryTitle": Column(str, nullable=False),
        "isAdult": Column(int, Check.eq(0), nullable=False),  # adult filtered out
        "startYear": Column(float, Check.greater_than_or_equal_to(1970), nullable=False),
        "runtimeMinutes": Column(float, Check.greater_than(0), nullable=False),
        "genres": Column(str, nullable=False),  # nulls filtered out
        "averageRating": Column(float, Check.in_range(0.0, 10.0), nullable=False),
        "numVotes": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
    },
    checks=[
        Check(lambda df: len(df) > 0, error="Filtered dataset must not be empty"),
    ],
    coerce=False,
    strict=False,
    name="FilteredDatasetSchema",
)


# =============================================================================
#  4) Feature-engineered schema (ready for model)
# =============================================================================

FeatureEngineeredSchema = pa.DataFrameSchema(
    columns={
        "tconst": Column(str, Check.str_startswith("tt"), nullable=False, unique=True),
        "primaryTitle": Column(str, nullable=False),
        "averageRating": Column(float, Check.in_range(0.0, 10.0), nullable=False),
        "numVotes": Column(int, Check.greater_than(0), nullable=False),
        "averageRating_scaled": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "numVotes_scaled": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "runtimeMinutes_scaled": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "startYear_scaled": Column(float, Check.in_range(0.0, 1.0), nullable=False),
    },
    checks=[
        Check(
            lambda df: any(c.startswith("genre_") for c in df.columns),
            error="At least one genre_ column is required after encoding",
        ),
        Check(lambda df: len(df) > 100, error="Dataset too small for meaningful recommendations"),
    ],
    coerce=False,
    strict=False,
    name="FeatureEngineeredSchema",
)


# =============================================================================
#  Validation runner
# =============================================================================


def validate_dataframe(
    df,
    schema: pa.DataFrameSchema,
    *,
    lazy: bool = True,
) -> pa.errors.SchemaErrors | None:
    """
    Validate a DataFrame against a Pandera schema.

    Args:
        df: The DataFrame to validate.
        schema: The Pandera schema to apply.
        lazy: If True, collect all errors before raising (recommended).

    Returns:
        None if validation passes.

    Raises:
        pandera.errors.SchemaErrors: If validation fails (contains all errors).
    """
    logger.info(f"Validating DataFrame ({len(df):,} rows) against {schema.name}...")
    try:
        schema.validate(df, lazy=lazy)
        logger.info(f"  ✅ {schema.name} — all checks passed")
        return None
    except pa.errors.SchemaErrors as exc:
        n_errors = len(exc.failure_cases)
        logger.warning(f"  ❌ {schema.name} — {n_errors} validation failure(s)")
        for _, row in exc.failure_cases.head(10).iterrows():
            logger.warning(f"     • {row.get('column', '?')}: {row.get('check', '?')}")
        raise
