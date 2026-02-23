"""
Unit Tests - Data Validation (Pandera)
========================================
Tests for the Pandera schemas defined in src/validation/schemas.py.
Ensures schemas correctly accept valid data and reject invalid data.
"""

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from src.validation.schemas import (
    FeatureEngineeredSchema,
    FilteredDatasetSchema,
    MergedDatasetSchema,
    TitleBasicsSchema,
    TitleRatingsSchema,
    validate_dataframe,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def valid_basics_df():
    """Valid title.basics DataFrame."""
    return pd.DataFrame(
        {
            "tconst": ["tt0000001", "tt0000002"],
            "titleType": ["movie", "tvSeries"],
            "primaryTitle": ["Test Movie", "Test Series"],
            "originalTitle": ["Test Movie", "Test Series"],
            "isAdult": [0, 0],
            "startYear": [2020.0, 2019.0],
            "endYear": [np.nan, 2021.0],
            "runtimeMinutes": [120.0, 45.0],
            "genres": ["Action,Drama", "Comedy"],
        }
    )


@pytest.fixture
def valid_ratings_df():
    """Valid title.ratings DataFrame."""
    return pd.DataFrame(
        {
            "tconst": ["tt0000001", "tt0000002"],
            "averageRating": [8.5, 7.2],
            "numVotes": [150000, 50000],
        }
    )


@pytest.fixture
def valid_merged_df():
    """Valid merged DataFrame (basics + ratings)."""
    return pd.DataFrame(
        {
            "tconst": ["tt0000001", "tt0000002"],
            "titleType": ["movie", "tvSeries"],
            "primaryTitle": ["Test Movie", "Test Series"],
            "isAdult": [0, 0],
            "startYear": [2020.0, 2019.0],
            "runtimeMinutes": [120.0, 45.0],
            "genres": ["Action,Drama", "Comedy"],
            "averageRating": [8.5, 7.2],
            "numVotes": [150000, 50000],
        }
    )


@pytest.fixture
def valid_filtered_df():
    """Valid filtered DataFrame."""
    return pd.DataFrame(
        {
            "tconst": ["tt0000001", "tt0000002"],
            "titleType": ["movie", "tvSeries"],
            "primaryTitle": ["Test Movie", "Test Series"],
            "isAdult": [0, 0],
            "startYear": [2020.0, 2019.0],
            "runtimeMinutes": [120.0, 45.0],
            "genres": ["Action,Drama", "Comedy"],
            "averageRating": [8.5, 7.2],
            "numVotes": [1500, 5000],
        }
    )


@pytest.fixture
def valid_feature_df():
    """Valid feature-engineered DataFrame."""
    return pd.DataFrame(
        {
            "tconst": [f"tt{i:07d}" for i in range(200)],
            "primaryTitle": [f"Movie {i}" for i in range(200)],
            "averageRating": np.random.uniform(1.0, 10.0, 200),
            "numVotes": np.random.randint(100, 500000, 200),
            "averageRating_scaled": np.random.uniform(0.0, 1.0, 200),
            "numVotes_scaled": np.random.uniform(0.0, 1.0, 200),
            "runtimeMinutes_scaled": np.random.uniform(0.0, 1.0, 200),
            "startYear_scaled": np.random.uniform(0.0, 1.0, 200),
            "genre_Action": np.random.randint(0, 2, 200),
            "genre_Drama": np.random.randint(0, 2, 200),
            "genre_Comedy": np.random.randint(0, 2, 200),
        }
    )


# ── TitleBasicsSchema ────────────────────────────────────────────────────


class TestTitleBasicsSchema:
    def test_valid_basics(self, valid_basics_df):
        """Valid basics data should pass."""
        TitleBasicsSchema.validate(valid_basics_df)

    def test_invalid_tconst_prefix(self, valid_basics_df):
        """tconst not starting with 'tt' should fail."""
        valid_basics_df.loc[0, "tconst"] = "nm0000001"
        with pytest.raises(pa.errors.SchemaError):
            TitleBasicsSchema.validate(valid_basics_df)

    def test_invalid_rating_out_of_range(self, valid_basics_df):
        """runtime < 0 should fail."""
        valid_basics_df.loc[0, "runtimeMinutes"] = -10.0
        with pytest.raises(pa.errors.SchemaError):
            TitleBasicsSchema.validate(valid_basics_df)

    def test_invalid_isadult_value(self, valid_basics_df):
        """isAdult not in {0, 1} should fail."""
        valid_basics_df.loc[0, "isAdult"] = 2
        with pytest.raises(pa.errors.SchemaError):
            TitleBasicsSchema.validate(valid_basics_df)


# ── TitleRatingsSchema ───────────────────────────────────────────────────


class TestTitleRatingsSchema:
    def test_valid_ratings(self, valid_ratings_df):
        """Valid ratings data should pass."""
        TitleRatingsSchema.validate(valid_ratings_df)

    def test_rating_out_of_range(self, valid_ratings_df):
        """Rating > 10 should fail."""
        valid_ratings_df.loc[0, "averageRating"] = 11.0
        with pytest.raises(pa.errors.SchemaError):
            TitleRatingsSchema.validate(valid_ratings_df)

    def test_negative_votes(self, valid_ratings_df):
        """Negative numVotes should fail."""
        valid_ratings_df.loc[0, "numVotes"] = -1
        with pytest.raises(pa.errors.SchemaError):
            TitleRatingsSchema.validate(valid_ratings_df)


# ── MergedDatasetSchema ──────────────────────────────────────────────────


class TestMergedDatasetSchema:
    def test_valid_merged(self, valid_merged_df):
        """Valid merged data should pass."""
        MergedDatasetSchema.validate(valid_merged_df)

    def test_duplicate_tconst(self, valid_merged_df):
        """Duplicate tconst should fail (uniqueness constraint)."""
        valid_merged_df.loc[1, "tconst"] = "tt0000001"
        with pytest.raises(pa.errors.SchemaError):
            MergedDatasetSchema.validate(valid_merged_df)

    def test_missing_column(self, valid_merged_df):
        """Missing required column should fail."""
        df = valid_merged_df.drop(columns=["averageRating"])
        with pytest.raises(pa.errors.SchemaError):
            MergedDatasetSchema.validate(df)


# ── FilteredDatasetSchema ────────────────────────────────────────────────


class TestFilteredDatasetSchema:
    def test_valid_filtered(self, valid_filtered_df):
        """Valid filtered data should pass."""
        FilteredDatasetSchema.validate(valid_filtered_df)

    def test_invalid_title_type(self, valid_filtered_df):
        """Unexpected title type (e.g., 'short') should fail."""
        valid_filtered_df.loc[0, "titleType"] = "short"
        with pytest.raises(pa.errors.SchemaError):
            FilteredDatasetSchema.validate(valid_filtered_df)

    def test_adult_content_rejected(self, valid_filtered_df):
        """isAdult=1 should fail after filtering."""
        valid_filtered_df.loc[0, "isAdult"] = 1
        with pytest.raises(pa.errors.SchemaError):
            FilteredDatasetSchema.validate(valid_filtered_df)

    def test_old_year_rejected(self, valid_filtered_df):
        """Year < 1970 should fail."""
        valid_filtered_df.loc[0, "startYear"] = 1950.0
        with pytest.raises(pa.errors.SchemaError):
            FilteredDatasetSchema.validate(valid_filtered_df)


# ── FeatureEngineeredSchema ──────────────────────────────────────────────


class TestFeatureEngineeredSchema:
    def test_valid_features(self, valid_feature_df):
        """Valid feature-engineered data should pass."""
        FeatureEngineeredSchema.validate(valid_feature_df)

    def test_scaled_out_of_range(self, valid_feature_df):
        """Scaled value > 1 should fail."""
        valid_feature_df.loc[0, "averageRating_scaled"] = 1.5
        with pytest.raises(pa.errors.SchemaError):
            FeatureEngineeredSchema.validate(valid_feature_df)

    def test_too_few_rows(self):
        """Dataset with < 100 rows should fail."""
        df = pd.DataFrame(
            {
                "tconst": [f"tt{i:07d}" for i in range(50)],
                "primaryTitle": [f"Movie {i}" for i in range(50)],
                "averageRating": np.random.uniform(1.0, 10.0, 50),
                "numVotes": np.random.randint(100, 500000, 50),
                "averageRating_scaled": np.random.uniform(0.0, 1.0, 50),
                "numVotes_scaled": np.random.uniform(0.0, 1.0, 50),
                "runtimeMinutes_scaled": np.random.uniform(0.0, 1.0, 50),
                "startYear_scaled": np.random.uniform(0.0, 1.0, 50),
                "genre_Action": np.random.randint(0, 2, 50),
            }
        )
        with pytest.raises(pa.errors.SchemaErrors):
            FeatureEngineeredSchema.validate(df, lazy=True)


# ── validate_dataframe helper ────────────────────────────────────────────


class TestValidateDataframe:
    def test_returns_none_on_success(self, valid_merged_df):
        """Should return None when validation passes."""
        result = validate_dataframe(valid_merged_df, MergedDatasetSchema)
        assert result is None

    def test_raises_on_failure(self, valid_merged_df):
        """Should raise SchemaErrors on invalid data."""
        valid_merged_df.loc[0, "averageRating"] = 99.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_dataframe(valid_merged_df, MergedDatasetSchema)
