"""
Unit Tests - Feature Engineering
=================================
Tests for the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    build_feature_matrix,
    create_text_features,
    encode_genres,
    filter_titles,
    scale_numerical_features,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame mimicking merged IMDb data."""
    return pd.DataFrame(
        {
            "tconst": ["tt001", "tt002", "tt003", "tt004", "tt005"],
            "titleType": ["movie", "movie", "tvSeries", "movie", "short"],
            "primaryTitle": [
                "Action Movie",
                "Comedy Film",
                "Drama Series",
                "Old Movie",
                "Short Film",
            ],
            "isAdult": [0, 0, 0, 0, 0],
            "startYear": [2020, 2019, 2021, 1950, 2022],
            "runtimeMinutes": [120, 90, 45, 100, 5],
            "genres": [
                "Action,Drama",
                "Comedy,Romance",
                "Drama,Thriller",
                "Western",
                "Short",
            ],
            "averageRating": [8.0, 7.5, 9.0, 6.5, 5.0],
            "numVotes": [50000, 30000, 100000, 200, 50],
        }
    )


class TestFilterTitles:
    """Tests for title filtering."""

    def test_filters_by_title_type(self, sample_dataframe):
        df = filter_titles(
            sample_dataframe,
            title_types=["movie"],
            min_votes=0,
            min_year=1900,
        )
        assert all(df["titleType"] == "movie")

    def test_filters_by_min_votes(self, sample_dataframe):
        df = filter_titles(
            sample_dataframe,
            min_votes=10000,
            min_year=1900,
        )
        assert all(df["numVotes"] >= 10000)

    def test_filters_by_min_year(self, sample_dataframe):
        df = filter_titles(
            sample_dataframe,
            min_votes=0,
            min_year=2000,
        )
        assert all(df["startYear"] >= 2000)

    def test_returns_dataframe(self, sample_dataframe):
        df = filter_titles(sample_dataframe, min_votes=0, min_year=1900)
        assert isinstance(df, pd.DataFrame)


class TestEncodeGenres:
    """Tests for genre encoding."""

    def test_creates_genre_columns(self, sample_dataframe):
        df, mlb = encode_genres(sample_dataframe)
        genre_cols = [c for c in df.columns if c.startswith("genre_")]
        assert len(genre_cols) > 0

    def test_binary_encoding(self, sample_dataframe):
        df, mlb = encode_genres(sample_dataframe)
        genre_cols = [c for c in df.columns if c.startswith("genre_")]
        for col in genre_cols:
            assert set(df[col].unique()).issubset({0, 1})

    def test_action_genre_present(self, sample_dataframe):
        df, mlb = encode_genres(sample_dataframe)
        assert "genre_Action" in df.columns
        # First row should have Action=1
        assert df["genre_Action"].iloc[0] == 1
        # Second row should have Action=0
        assert df["genre_Action"].iloc[1] == 0


class TestScaleNumerical:
    """Tests for numerical scaling."""

    def test_scaled_range(self, sample_dataframe):
        df, scalers = scale_numerical_features(sample_dataframe)
        for col in ["averageRating_scaled", "numVotes_scaled"]:
            if col in df.columns:
                assert df[col].min() >= 0.0
                assert df[col].max() <= 1.0

    def test_returns_scalers(self, sample_dataframe):
        df, scalers = scale_numerical_features(sample_dataframe)
        assert isinstance(scalers, dict)
        assert "averageRating" in scalers


class TestBuildFeatureMatrix:
    """Tests for feature matrix construction."""

    def test_output_shape(self, sample_dataframe):
        df, _ = encode_genres(sample_dataframe)
        df, _ = scale_numerical_features(df)
        matrix, names = build_feature_matrix(df)
        assert matrix.shape[0] == len(df)
        assert matrix.shape[1] == len(names)
        assert matrix.dtype == np.float32

    def test_no_nan_values(self, sample_dataframe):
        df, _ = encode_genres(sample_dataframe)
        df, _ = scale_numerical_features(df)
        matrix, _ = build_feature_matrix(df)
        assert not np.isnan(matrix).any()


class TestTextFeatures:
    """Tests for text feature creation."""

    def test_creates_text_column(self, sample_dataframe):
        df = create_text_features(sample_dataframe)
        assert "text_feature" in df.columns
        assert len(df["text_feature"].iloc[0]) > 0
