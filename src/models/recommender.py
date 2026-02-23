"""
Recommendation Model
=====================
Content-based recommendation engine using cosine similarity.
Supports both structured feature similarity and semantic embeddings.
"""

import logging
import pickle
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentRecommender:
    """
    Content-based recommendation engine.

    Computes cosine similarity over structured features (genres, rating,
    votes, runtime, year) for efficient top-k retrieval.
    """

    def __init__(self):
        self.feature_matrix: np.ndarray | None = None
        self.title_index: dict[str, int] | None = None
        self.index_to_tconst: dict[int, str] | None = None
        self.metadata: dict | None = None
        self._is_fitted = False

    def fit(
        self,
        feature_matrix: np.ndarray,
        tconst_ids: list[str],
        titles: list[str],
        metadata: dict | None = None,
    ) -> "ContentRecommender":
        """
        Fit the recommender with precomputed feature matrix.

        Args:
            feature_matrix: Numpy array of shape (n_titles, n_features).
            tconst_ids: List of IMDb tconst identifiers.
            titles: List of primary titles (for lookup).
            metadata: Optional metadata dict for serialization.
        """
        self.feature_matrix = feature_matrix.astype(np.float32)
        self.tconst_ids = tconst_ids
        self.titles = titles

        # Build lookup indices: title -> index (case-insensitive)
        self.title_index = {}
        for i, title in enumerate(titles):
            key = title.strip().lower()
            # Keep first occurrence (typically has more votes)
            if key not in self.title_index:
                self.title_index[key] = i

        # Build tconst -> index lookup for unique identification
        self.tconst_index = {tc: i for i, tc in enumerate(tconst_ids)}

        self.index_to_tconst = {i: tc for i, tc in enumerate(tconst_ids)}
        self.metadata = metadata or {}
        self._is_fitted = True

        logger.info(
            f"Recommender fitted: {len(tconst_ids):,} titles, "
            f"{feature_matrix.shape[1]} features"
        )
        return self

    def _validate_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def find_tconst_index(self, tconst: str) -> int | None:
        """Find the index of a title by its IMDb tconst ID (exact, unique)."""
        self._validate_fitted()
        return self.tconst_index.get(tconst)

    def find_title_index(self, title: str) -> int | None:
        """Find the index of a title (case-insensitive fuzzy match)."""
        self._validate_fitted()

        key = title.strip().lower()

        # Exact match
        if key in self.title_index:
            return self.title_index[key]

        # Partial match (prefix)
        matches = [(k, idx) for k, idx in self.title_index.items() if key in k or k in key]

        if matches:
            # Return the best match (shortest key that contains query)
            matches.sort(key=lambda x: len(x[0]))
            return matches[0][1]

        return None

    def recommend(
        self,
        title: str,
        top_k: int = 10,
        tconst: str | None = None,
    ) -> list[dict]:
        """
        Get top-k recommendations for a given title.

        Args:
            title: Movie/series title to find recommendations for.
            top_k: Number of recommendations to return.
            tconst: Optional IMDb tconst ID for exact identification.
                    If provided, takes priority over title matching.

        Returns:
            List of recommendation dicts with title, score, and metadata.
        """
        self._validate_fitted()

        idx = None
        # Prefer tconst lookup (unique, no ambiguity)
        if tconst:
            idx = self.find_tconst_index(tconst)
        # Fallback to title lookup
        if idx is None:
            idx = self.find_title_index(title)
        if idx is None:
            raise ValueError(f"Title not found: '{title}'")

        return self._recommend_by_index(idx, top_k)

    def _recommend_by_index(self, idx: int, top_k: int) -> list[dict]:
        """Internal recommendation by index."""
        # Compute similarity of this title against all others
        query_vector = self.feature_matrix[idx : idx + 1]
        similarities = cosine_similarity(query_vector, self.feature_matrix).flatten()

        # Get top-k+1 (excluding self)
        top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

        results = []
        for rank, i in enumerate(top_indices, 1):
            results.append(
                {
                    "rank": rank,
                    "tconst": self.tconst_ids[i],
                    "title": self.titles[i],
                    "similarity_score": round(float(similarities[i]), 4),
                }
            )

        return results

    def search_titles(self, query: str, limit: int = 20) -> list[dict]:
        """
        Search for titles matching a query string.

        Uses a multi-tier matching strategy:
          1. Exact prefix match (title starts with query)
          2. Substring match (query appears anywhere in title)
          3. Fuzzy match (handles typos via SequenceMatcher)

        Results are sorted by relevance: prefix > substring > fuzzy.

        Args:
            query: Search query (partial title match, typo-tolerant).
            limit: Maximum results to return.

        Returns:
            List of matching title dicts sorted by relevance.
        """
        self._validate_fitted()
        query_lower = query.strip().lower()

        if not query_lower:
            return []

        prefix_matches = []  # tier 1: title starts with query
        contains_matches = []  # tier 2: query is a substring
        fuzzy_matches = []  # tier 3: fuzzy / approximate

        # Fuzzy threshold — lower = more permissive
        # Adapt based on query length (short queries need stricter match)
        fuzzy_threshold = 0.55 if len(query_lower) >= 5 else 0.7

        for i, title in enumerate(self.titles):
            title_lower = title.lower()

            if title_lower.startswith(query_lower):
                prefix_matches.append((i, title, 1.0))
            elif query_lower in title_lower:
                contains_matches.append((i, title, 0.9))
            elif len(query_lower) >= 3:
                # Fuzzy: compare against each word and full title
                ratio = SequenceMatcher(None, query_lower, title_lower).ratio()
                # Also check individual words for partial matches
                words = title_lower.split()
                word_ratios = [SequenceMatcher(None, query_lower, w).ratio() for w in words]
                best_ratio = max(ratio, max(word_ratios) if word_ratios else 0)
                if best_ratio >= fuzzy_threshold:
                    fuzzy_matches.append((i, title, best_ratio))

        # Sort fuzzy by score descending
        fuzzy_matches.sort(key=lambda x: x[2], reverse=True)

        # Combine tiers in priority order
        combined = prefix_matches + contains_matches + fuzzy_matches

        results = []
        seen = set()
        for idx, title, _score in combined:
            if idx not in seen:
                seen.add(idx)
                results.append(
                    {
                        "index": idx,
                        "tconst": self.tconst_ids[idx],
                        "title": title,
                    }
                )
                if len(results) >= limit:
                    break

        return results

    def save(self, path: Path) -> None:
        """Save the fitted model to disk."""
        self._validate_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "feature_matrix": self.feature_matrix,
            "tconst_ids": self.tconst_ids,
            "titles": self.titles,
            "metadata": self.metadata,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = path.stat().st_size / 1e6
        logger.info(f"Model saved to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "ContentRecommender":
        """Load a fitted model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls()
        model.fit(
            feature_matrix=state["feature_matrix"],
            tconst_ids=state["tconst_ids"],
            titles=state["titles"],
            metadata=state.get("metadata"),
        )

        logger.info(f"Model loaded from {path}")
        return model
