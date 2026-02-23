"""
IMDb Dataset Downloader
=======================
Robust, idempotent downloader for IMDb Non-Commercial datasets.
Supports caching, checksum verification, and progress tracking.
"""

import gzip
import hashlib
import logging
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Official IMDb dataset URLs
IMDB_BASE_URL = "https://datasets.imdbws.com"
DATASET_FILES = {
    "title.basics": "title.basics.tsv.gz",
    "title.ratings": "title.ratings.tsv.gz",
    "title.principals": "title.principals.tsv.gz",
}

DEFAULT_RAW_DIR = Path("data/raw")


def _compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file for integrity verification."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def _download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1024 * 1024,
    timeout: int = 120,
) -> Path:
    """
    Download a single file with progress bar and streaming.

    Args:
        url: URL to download from.
        dest: Destination file path.
        chunk_size: Download chunk size in bytes.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded file.
    """
    logger.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            ncols=80,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _decompress_gz(gz_path: Path, output_path: Path) -> Path:
    """Decompress a gzip file to TSV."""
    logger.info(f"Decompressing {gz_path.name} -> {output_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"Decompressed {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def download_dataset(
    dataset_key: str,
    raw_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download and decompress a single IMDb dataset file.

    Idempotent: skips download if file already exists (unless force=True).

    Args:
        dataset_key: Key from DATASET_FILES (e.g., 'title.basics').
        raw_dir: Directory to store raw data files.
        force: Force re-download even if file exists.

    Returns:
        Path to the decompressed TSV file.
    """
    if dataset_key not in DATASET_FILES:
        raise ValueError(
            f"Unknown dataset: {dataset_key}. " f"Available: {list(DATASET_FILES.keys())}"
        )

    raw_dir = Path(raw_dir or DEFAULT_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = DATASET_FILES[dataset_key]
    gz_path = raw_dir / filename
    tsv_path = raw_dir / filename.replace(".gz", "")

    # Idempotent check
    if tsv_path.exists() and not force:
        logger.info(f"Cached: {tsv_path} already exists. Skipping download.")
        return tsv_path

    # Download
    url = f"{IMDB_BASE_URL}/{filename}"
    _download_file(url, gz_path)

    # Store checksum for versioning
    checksum = _compute_md5(gz_path)
    checksum_file = raw_dir / f"{filename}.md5"
    checksum_file.write_text(f"{checksum}  {filename}\n")
    logger.info(f"Checksum: {checksum}")

    # Decompress
    _decompress_gz(gz_path, tsv_path)

    # Clean up compressed file to save space
    gz_path.unlink()
    logger.info(f"Removed compressed file: {gz_path.name}")

    return tsv_path


def download_all_datasets(
    raw_dir: Path | None = None,
    force: bool = False,
    include_principals: bool = False,
) -> dict[str, Path]:
    """
    Download all required IMDb datasets.

    Args:
        raw_dir: Directory to store raw data files.
        force: Force re-download even if files exist.
        include_principals: Whether to include title.principals dataset.

    Returns:
        Dictionary mapping dataset keys to file paths.
    """
    datasets_to_download = ["title.basics", "title.ratings"]
    if include_principals:
        datasets_to_download.append("title.principals")

    results = {}
    for key in datasets_to_download:
        try:
            path = download_dataset(key, raw_dir=raw_dir, force=force)
            results[key] = path
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            raise

    logger.info(f"Successfully downloaded {len(results)} datasets")
    return results
