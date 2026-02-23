"""
Unit Tests - Recommendation Model
===================================
Tests for the ContentRecommender model.
"""

from pathlib import Path

import numpy as np
import pytest

from src.models.recommender import ContentRecommender


@pytest.fixture
def fitted_model():
    """Create a fitted ContentRecommender for testing."""
    np.random.seed(42)
    n_titles = 100
    n_features = 10

    feature_matrix = np.random.rand(n_titles, n_features).astype(np.float32)
    tconst_ids = [f"tt{i:07d}" for i in range(n_titles)]
    titles = [f"Test Movie {i}" for i in range(n_titles)]

    # Make some titles have known names for search testing
    titles[0] = "Inception"
    titles[1] = "The Matrix"
    titles[2] = "Interstellar"
    titles[3] = "The Dark Knight"

    model = ContentRecommender()
    model.fit(feature_matrix, tconst_ids, titles)
    return model


class TestContentRecommender:
    """Tests for the ContentRecommender."""

    def test_fit(self, fitted_model):
        assert fitted_model._is_fitted
        assert len(fitted_model.titles) == 100

    def test_recommend_returns_correct_count(self, fitted_model):
        recs = fitted_model.recommend("Inception", top_k=5)
        assert len(recs) == 5

    def test_recommend_excludes_self(self, fitted_model):
        recs = fitted_model.recommend("Inception", top_k=5)
        for rec in recs:
            assert rec["title"] != "Inception"

    def test_recommend_has_required_keys(self, fitted_model):
        recs = fitted_model.recommend("Inception", top_k=3)
        for rec in recs:
            assert "rank" in rec
            assert "tconst" in rec
            assert "title" in rec
            assert "similarity_score" in rec

    def test_recommend_sorted_by_similarity(self, fitted_model):
        recs = fitted_model.recommend("Inception", top_k=10)
        scores = [r["similarity_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_unknown_title_raises(self, fitted_model):
        with pytest.raises(ValueError, match="Title not found"):
            fitted_model.recommend("Nonexistent Movie 12345")

    def test_search_titles(self, fitted_model):
        results = fitted_model.search_titles("Matrix")
        assert len(results) > 0
        assert results[0]["title"] == "The Matrix"

    def test_save_and_load(self, fitted_model, tmp_path):
        model_path = tmp_path / "model.pkl"
        fitted_model.save(model_path)
        assert model_path.exists()

        loaded = ContentRecommender.load(model_path)
        assert loaded._is_fitted
        assert len(loaded.titles) == len(fitted_model.titles)

        # Same recommendations
        recs_original = fitted_model.recommend("Inception", top_k=5)
        recs_loaded = loaded.recommend("Inception", top_k=5)
        assert recs_original == recs_loaded

    def test_unfitted_model_raises(self):
        model = ContentRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend("Inception")

    def test_case_insensitive_search(self, fitted_model):
        idx1 = fitted_model.find_title_index("inception")
        idx2 = fitted_model.find_title_index("INCEPTION")
        idx3 = fitted_model.find_title_index("Inception")
        assert idx1 == idx2 == idx3

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            ContentRecommender.load(Path("/nonexistent/model.pkl"))
