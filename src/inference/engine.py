"""
Inference Engine
=================
High-level inference interface for the recommendation system.
Loads trained artifacts and exposes a clean API.
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.recommender import ContentRecommender

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("data/processed")


class RecommendationEngine:
    """
    Production inference engine.

    Loads trained model artifacts and metadata to serve recommendations.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
    ):
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.model: ContentRecommender | None = None
        self.title_metadata: pd.DataFrame | None = None
        self._is_loaded = False

    def load(self) -> "RecommendationEngine":
        """Load all trained artifacts."""
        model_path = self.model_dir / "recommender_model.pkl"
        metadata_path = self.model_dir / "title_metadata.parquet"

        # Load model
        self.model = ContentRecommender.load(model_path)

        # Load title metadata for enriched responses
        if metadata_path.exists():
            self.title_metadata = pd.read_parquet(metadata_path)
            logger.info(f"Loaded metadata: {len(self.title_metadata):,} titles")
        else:
            logger.warning("Title metadata not found, responses will be basic")

        self._is_loaded = True
        logger.info("Recommendation engine loaded successfully")
        return self

    def _validate_loaded(self):
        if not self._is_loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

    def recommend(
        self,
        title: str,
        top_k: int = 10,
        tconst: str | None = None,
    ) -> dict:
        """
        Get enriched recommendations for a title.

        Args:
            title: Movie/series title.
            top_k: Number of recommendations.
            tconst: Optional IMDb tconst ID for exact identification.

        Returns:
            Dict with query info and enriched recommendations.
        """
        self._validate_loaded()

        # Get base recommendations (tconst takes priority)
        raw_recs = self.model.recommend(title, top_k=top_k, tconst=tconst)

        # Find the query title index for metadata
        query_idx = None
        if tconst:
            query_idx = self.model.find_tconst_index(tconst)
        if query_idx is None:
            query_idx = self.model.find_title_index(title)
        query_info = self._enrich_title(query_idx) if query_idx is not None else {"title": title}

        # Enrich recommendations with metadata
        enriched = []
        for rec in raw_recs:
            tconst = rec["tconst"]
            enriched_rec = {
                "rank": rec["rank"],
                "tconst": tconst,
                "title": rec["title"],
                "similarity_score": rec["similarity_score"],
            }

            # Add metadata if available
            if self.title_metadata is not None:
                meta_row = self.title_metadata[self.title_metadata["tconst"] == tconst]
                if not meta_row.empty:
                    row = meta_row.iloc[0]
                    enriched_rec.update(
                        {
                            "year": (
                                int(row["startYear"]) if pd.notna(row.get("startYear")) else None
                            ),
                            "rating": (
                                float(row["averageRating"])
                                if pd.notna(row.get("averageRating"))
                                else None
                            ),
                            "votes": (
                                int(row["numVotes"]) if pd.notna(row.get("numVotes")) else None
                            ),
                            "genres": row.get("genres", ""),
                            "runtime_minutes": (
                                int(row["runtimeMinutes"])
                                if pd.notna(row.get("runtimeMinutes"))
                                else None
                            ),
                            "title_type": row.get("titleType", ""),
                        }
                    )

            enriched.append(enriched_rec)

        return {
            "query": query_info,
            "top_k": top_k,
            "recommendations": enriched,
        }

    def _enrich_title(self, idx: int) -> dict:
        """Get enriched info for a title by model index."""
        tconst = self.model.tconst_ids[idx]
        info = {
            "tconst": tconst,
            "title": self.model.titles[idx],
        }

        if self.title_metadata is not None:
            meta_row = self.title_metadata[self.title_metadata["tconst"] == tconst]
            if not meta_row.empty:
                row = meta_row.iloc[0]
                info.update(
                    {
                        "year": int(row["startYear"]) if pd.notna(row.get("startYear")) else None,
                        "rating": (
                            float(row["averageRating"])
                            if pd.notna(row.get("averageRating"))
                            else None
                        ),
                        "votes": int(row["numVotes"]) if pd.notna(row.get("numVotes")) else None,
                        "genres": row.get("genres", ""),
                        "runtime_minutes": (
                            int(row["runtimeMinutes"])
                            if pd.notna(row.get("runtimeMinutes"))
                            else None
                        ),
                    }
                )

        return info

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """
        Search for titles matching a query.

        Returns enriched results with metadata.
        """
        self._validate_loaded()
        matches = self.model.search_titles(query, limit=limit)

        enriched = []
        for match in matches:
            info = {
                "tconst": match["tconst"],
                "title": match["title"],
            }
            if self.title_metadata is not None:
                meta_row = self.title_metadata[self.title_metadata["tconst"] == match["tconst"]]
                if not meta_row.empty:
                    row = meta_row.iloc[0]
                    info.update(
                        {
                            "year": (
                                int(row["startYear"]) if pd.notna(row.get("startYear")) else None
                            ),
                            "rating": (
                                float(row["averageRating"])
                                if pd.notna(row.get("averageRating"))
                                else None
                            ),
                            "genres": row.get("genres", ""),
                        }
                    )
            enriched.append(info)

        return enriched

    def get_available_titles_count(self) -> int:
        """Return the number of titles in the model."""
        self._validate_loaded()
        return len(self.model.titles)
