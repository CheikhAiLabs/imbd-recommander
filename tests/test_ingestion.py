"""
Unit Tests - Data Ingestion
============================
Tests for the downloader and loader modules.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.ingestion.downloader import (
    DATASET_FILES,
    _compute_md5,
    download_dataset,
)
from src.ingestion.loader import (
    load_title_basics,
    load_title_ratings,
)


class TestDownloader:
    """Tests for the data downloader module."""

    def test_dataset_files_contains_required_keys(self):
        """Verify all required dataset keys are defined."""
        assert "title.basics" in DATASET_FILES
        assert "title.ratings" in DATASET_FILES
        assert "title.principals" in DATASET_FILES

    def test_unknown_dataset_raises_error(self):
        """Unknown dataset key should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            download_dataset("fake.dataset")

    def test_compute_md5(self, tmp_path):
        """MD5 computation should return consistent hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        hash1 = _compute_md5(test_file)
        hash2 = _compute_md5(test_file)
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_idempotent_skip(self, tmp_path):
        """Should skip download if file already exists."""
        # Create a fake TSV file
        tsv_file = tmp_path / "title.basics.tsv"
        tsv_file.write_text("tconst\ttitleType\n")

        with patch("src.ingestion.downloader._download_file") as mock_dl:
            result = download_dataset("title.basics", raw_dir=tmp_path, force=False)
            mock_dl.assert_not_called()
            assert result == tsv_file


class TestLoader:
    """Tests for the data loader module."""

    def test_load_title_basics(self, tmp_path):
        """Should correctly load and parse title.basics TSV."""
        content = (
            "tconst\ttitleType\tprimaryTitle\toriginalTitle\t"
            "isAdult\tstartYear\tendYear\truntimeMinutes\tgenres\n"
            "tt0000001\tmovie\tTest Movie\tTest Movie\t"
            "0\t2020\t\\N\t120\tAction,Drama\n"
            "tt0000002\ttvSeries\tTest Series\tTest Series\t"
            "0\t2019\t2021\t45\tComedy\n"
        )
        filepath = tmp_path / "title.basics.tsv"
        filepath.write_text(content)

        df = load_title_basics(filepath)
        assert len(df) == 2
        assert df["tconst"].iloc[0] == "tt0000001"
        assert df["startYear"].iloc[0] == 2020.0
        assert df["runtimeMinutes"].iloc[0] == 120.0
        assert pd.isna(df["endYear"].iloc[0])  # \\N -> NaN
        assert df["genres"].iloc[0] == "Action,Drama"

    def test_load_title_ratings(self, tmp_path):
        """Should correctly load title.ratings TSV."""
        content = (
            "tconst\taverageRating\tnumVotes\n" "tt0000001\t8.5\t150000\n" "tt0000002\t7.2\t50000\n"
        )
        filepath = tmp_path / "title.ratings.tsv"
        filepath.write_text(content)

        df = load_title_ratings(filepath)
        assert len(df) == 2
        assert df["averageRating"].iloc[0] == 8.5
        assert df["numVotes"].iloc[1] == 50000
