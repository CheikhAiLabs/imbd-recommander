"""
Unit Tests - API
=================
Tests for the FastAPI recommendation service.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_engine():
    """Create a mock recommendation engine."""
    mock = MagicMock()
    mock._is_loaded = True
    mock.get_available_titles_count.return_value = 1000
    mock.recommend.return_value = {
        "query": {
            "tconst": "tt1375666",
            "title": "Inception",
            "year": 2010,
            "rating": 8.8,
            "votes": 2300000,
            "genres": "Action,Adventure,Sci-Fi",
            "runtime_minutes": 148,
        },
        "top_k": 5,
        "recommendations": [
            {
                "rank": 1,
                "tconst": "tt0816692",
                "title": "Interstellar",
                "similarity_score": 0.95,
                "year": 2014,
                "rating": 8.7,
                "votes": 1900000,
                "genres": "Adventure,Drama,Sci-Fi",
                "runtime_minutes": 169,
                "title_type": "movie",
            },
        ],
    }
    mock.search.return_value = [
        {
            "tconst": "tt1375666",
            "title": "Inception",
            "year": 2010,
            "rating": 8.8,
            "genres": "Action,Adventure,Sci-Fi",
        }
    ]
    return mock


@pytest.fixture
def client(mock_engine):
    """Create a test client with mocked engine."""
    with patch("api.main.engine", mock_engine):
        from api.main import app

        client = TestClient(app)
        yield client


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestRecommendEndpoint:
    def test_recommend_success(self, client):
        response = client.post(
            "/recommend",
            json={"title": "Inception", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert data["recommendations"][0]["title"] == "Interstellar"

    def test_recommend_missing_title(self, client):
        response = client.post(
            "/recommend",
            json={"top_k": 5},
        )
        assert response.status_code == 422  # Validation error

    def test_recommend_invalid_top_k(self, client):
        response = client.post(
            "/recommend",
            json={"title": "Inception", "top_k": 0},
        )
        assert response.status_code == 422

    def test_recommend_title_not_found(self, client, mock_engine):
        mock_engine.recommend.side_effect = ValueError("Title not found: 'xyz'")
        response = client.post(
            "/recommend",
            json={"title": "xyz", "top_k": 5},
        )
        assert response.status_code == 404


class TestSearchEndpoint:
    def test_search_success(self, client):
        response = client.get("/search", params={"q": "Inception"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert data[0]["title"] == "Inception"

    def test_search_empty_query(self, client):
        response = client.get("/search", params={"q": ""})
        assert response.status_code == 422


class TestRootEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
