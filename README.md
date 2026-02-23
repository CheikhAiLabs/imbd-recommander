<p align="center">
  <img src="https://img.icons8.com/color/96/movie-projector.png" alt="Movie Projector" width="96" height="96"/>
</p>

<h1 align="center">🎬 IMDb Recommender System</h1>

<p align="center">
  <strong>A production-grade, content-based Movie & Series Recommendation Engine</strong><br/>
  <em>Powered by official IMDb Non-Commercial Datasets with full MLOps observability</em>
</p>

<p align="center">
  <a href="https://github.com/CheikhAiLabs/imbd-recommander/actions/workflows/ci.yml">
    <img src="https://github.com/CheikhAiLabs/imbd-recommander/actions/workflows/ci.yml/badge.svg" alt="CI/CD"/>
  </a>
  <a href="https://www.python.org/downloads/release/python-3120/">
    <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12+"/>
  </a>
  <a href="https://fastapi.tiangolo.com">
    <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/>
  </a>
  <a href="https://mlflow.org">
    <img src="https://img.shields.io/badge/MLflow-3.10+-0194E2?style=flat&logo=mlflow&logoColor=white" alt="MLflow"/>
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat" alt="License: MIT"/>
  </a>
</p>

<p align="center">
  <a href="https://prometheus.io">
    <img src="https://img.shields.io/badge/Prometheus-E6522C?style=flat&logo=prometheus&logoColor=white" alt="Prometheus"/>
  </a>
  <a href="https://grafana.com">
    <img src="https://img.shields.io/badge/Grafana-F46800?style=flat&logo=grafana&logoColor=white" alt="Grafana"/>
  </a>
  <a href="https://streamlit.io">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit"/>
  </a>
  <a href="https://scikit-learn.org">
    <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/code%20style-ruff-261230?style=flat&logo=ruff&logoColor=white" alt="Ruff"/>
  </a>
  <a href="https://black.readthedocs.io">
    <img src="https://img.shields.io/badge/formatter-black-000000?style=flat&logo=python&logoColor=white" alt="Black"/>
  </a>
  <a href="https://pandera.readthedocs.io">
    <img src="https://img.shields.io/badge/validation-Pandera-163B6E?style=flat&logoColor=white" alt="Pandera"/>
  </a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-monitoring--observability">Monitoring</a> •
  <a href="#%EF%B8%8F-configuration">Config</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📖 Table of Contents

- [📖 Table of Contents](#-table-of-contents)
- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
  - [System Architecture Diagram](#system-architecture-diagram)
  - [Tech Stack \& Official Docs](#tech-stack--official-docs)
- [📁 Project Structure](#-project-structure)
  - [🔑 Key Files Explained](#-key-files-explained)
- [🚀 Quick Start](#-quick-start)
  - [Option A — Docker (Recommended)](#option-a--docker-recommended)
  - [Option B — Local Development](#option-b--local-development)
- [🧠 How It Works](#-how-it-works)
  - [1️⃣ Data Ingestion](#1️⃣-data-ingestion)
  - [2️⃣ Feature Engineering](#2️⃣-feature-engineering)
  - [3️⃣ Model Training](#3️⃣-model-training)
  - [4️⃣ Inference Engine](#4️⃣-inference-engine)
- [🎯 Training Pipeline](#-training-pipeline)
  - [Running Training](#running-training)
  - [What Gets Logged to MLflow](#what-gets-logged-to-mlflow)
  - [Training Output Artifacts](#training-output-artifacts)
- [📡 API Reference](#-api-reference)
  - [`POST /recommend`](#post-recommend)
  - [`GET /search`](#get-search)
  - [`GET /health`](#get-health)
  - [`GET /metrics`](#get-metrics)
- [🖥️ Web Interface](#️-web-interface)
- [📊 Monitoring \& Observability](#-monitoring--observability)
  - [Prometheus Metrics](#prometheus-metrics)
  - [Grafana Dashboard](#grafana-dashboard)
  - [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [⚙️ Configuration](#️-configuration)
- [🐳 Docker Deployment](#-docker-deployment)
  - [Services \& Ports](#services--ports)
  - [Docker Commands](#docker-commands)
  - [Volumes](#volumes)
- [🧪 Testing](#-testing)
- [🛠️ Development](#️-development)
  - [Available Make Targets](#available-make-targets)
  - [Code Quality](#code-quality)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

**IMDb Recommender** is a **fully production-ready**, end-to-end content-based recommendation system built on top of the [official IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/). It identifies similar movies and TV series using structured features—genres, ratings, vote counts, runtime, and release year—and serves them through a beautiful web interface backed by a high-performance REST API.

This isn't just a toy project. It includes:

- 🔄 **Reproducible ML pipeline** — from raw data download to trained model in one command
- �️ **Data validation** — Pandera schemas validate every pipeline stage (raw → merged → filtered → model-ready)
- �📈 **Full MLOps stack** — MLflow v3.10 experiment tracking with 60+ metrics, 8 distribution charts, 6 data tables, model registry, and API inference tracing
- 🔍 **Production API** — FastAPI with Prometheus instrumentation, health checks, structured logging
- 🎨 **Modern UI** — Streamlit app with dark theme, fuzzy search, polished movie cards, and IMDb-gold design language
- 📊 **Observability** — Grafana dashboards with 20+ panels across 6 monitoring sections
- 🐳 **One-click deployment** — Docker Compose with 5 services, health checks, auto-provisioned dashboards

> **79,596 titles** indexed • **30 features** per title • **Sub-50ms** recommendation latency

---

## ✨ Features

| Category | Feature | Description |
|:--------:|---------|-------------|
| 🧠 | **Content-Based Filtering** | Cosine similarity over 30 structured features (4 scaled numerical + 26 one-hot genre features) |
| 📥 | **Automatic Data Pipeline** | Idempotent download of IMDb datasets (~1.1 GB) with caching, checksums, and progress bars |
| ⚙️ | **Feature Engineering** | Automated filtering, genre encoding ([`MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html)), numerical scaling ([`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)) |
| 🚀 | **REST API** | [FastAPI](https://fastapi.tiangolo.com) with auto-generated OpenAPI docs, request tracing, X-Request-ID headers |
| 🎨 | **Web Interface** | [Streamlit](https://docs.streamlit.io) dark-themed UI with fuzzy search, movie cards with IMDb links, genre tags, similarity badges |
| 🔍 | **Fuzzy Search** | Typo-tolerant title search using multi-tier matching (prefix → substring → fuzzy via `SequenceMatcher`) |
| 📈 | **Experiment Tracking** | [MLflow](https://mlflow.org/docs/latest/index.html) v3.10 integration: 60+ metrics, 8 charts, 6 tables, model registry, dataset logging, inference tracing |
| 📊 | **Monitoring** | [Prometheus](https://prometheus.io/docs/) metrics + [Grafana](https://grafana.com/docs/) dashboards with 20+ pre-built panels |
| 🐳 | **Containerized** | Full [Docker Compose](https://docs.docker.com/compose/) stack with 5 services, health checks, named volumes |
| 🛡️ | **Data Validation** | [Pandera](https://pandera.readthedocs.io/) schemas enforce column types, value ranges, and constraints at every pipeline stage |
| 🧪 | **Tested** | [pytest](https://docs.pytest.org/) test suite with 56 tests covering ingestion, features, model, API, and validation |
| 🔧 | **Code Quality** | [Ruff](https://docs.astral.sh/ruff/) linting + [Black](https://black.readthedocs.io/) formatting, consistent style |
| 🔁 | **CI/CD** | [GitHub Actions](https://docs.github.com/en/actions) pipeline: lint → test → Docker build verification |

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IMDb Recommender System                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │   Streamlit   │───▶│   FastAPI + 🔥   │◀───│   Prometheus       │    │
│  │   UI :9877    │    │   API    :9876   │    │   Scraper  :9879   │    │
│  └──────────────┘    └──────┬───────────┘    └────────┬───────────┘    │
│                             │                         │                 │
│                             ▼                         ▼                 │
│                   ┌─────────────────┐       ┌─────────────────┐        │
│                   │  Trained Model  │       │    Grafana       │        │
│                   │  (cosine sim)   │       │  Dashboards      │        │
│                   │   79,596 titles │       │    :9880          │        │
│                   └─────────────────┘       └─────────────────┘        │
│                             │                                           │
│                             ▼                                           │
│                   ┌─────────────────┐                                   │
│                   │     MLflow      │                                   │
│                   │  Tracking :9878 │                                   │
│                   │  60+ metrics    │                                   │
│                   └─────────────────┘                                   │
│                                                                         │
├──── Training Pipeline ──────────────────────────────────────────────────┤
│                                                                         │
│  IMDb Datasets ──▶ Load & Merge ──▶ Filter ──▶ Encode ──▶ Scale        │
│  (~1.1 GB TSV)     (pandas)         (votes,    (genres    (MinMax       │
│                                      year,      MLB)      Scaler)      │
│                                      type)                              │
│                         ──▶ Fit Model ──▶ Save Artifacts               │
│                             (cosine       (.pkl + .parquet)             │
│                              similarity)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tech Stack & Official Docs

| Layer | Technology | Version | Purpose | Docs |
|:-----:|-----------|:-------:|---------|:----:|
| 🐍 | **Python** | 3.12+ | Core language | [docs.python.org](https://docs.python.org/3/) |
| 🔬 | **scikit-learn** | ≥1.3 | Cosine similarity, encoders, scalers | [scikit-learn.org](https://scikit-learn.org/stable/) |
| 🐼 | **pandas** | ≥2.0 | Data manipulation & TSV parsing | [pandas.pydata.org](https://pandas.pydata.org/docs/) |
| 🔢 | **NumPy** | ≥1.24 | Numerical computations, feature matrices | [numpy.org](https://numpy.org/doc/) |
| 🚀 | **FastAPI** | ≥0.109 | REST API with auto docs, async support | [fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| 🦄 | **Uvicorn** | ≥0.27 | ASGI server for FastAPI | [uvicorn.org](https://www.uvicorn.org) |
| 📐 | **Pydantic** | ≥2.5 | Request/response validation & serialization | [docs.pydantic.dev](https://docs.pydantic.dev/latest/) |
| 🎨 | **Streamlit** | ≥1.30 | Interactive web UI | [docs.streamlit.io](https://docs.streamlit.io) |
| 📈 | **MLflow** | ≥3.10 | Experiment tracking, artifact logging | [mlflow.org/docs](https://mlflow.org/docs/latest/index.html) |
| 🔥 | **Prometheus** | v3.2 | Metrics collection & alerting | [prometheus.io/docs](https://prometheus.io/docs/introduction/overview/) |
| 📊 | **Grafana** | v11.5 | Metrics visualization & dashboards | [grafana.com/docs](https://grafana.com/docs/grafana/latest/) |
| 🐳 | **Docker** | Compose v2 | Container orchestration | [docs.docker.com](https://docs.docker.com/compose/) |
| 🏹 | **PyArrow** | ≥14.0 | Parquet I/O for metadata | [arrow.apache.org](https://arrow.apache.org/docs/python/) |
| 📦 | **tqdm** | ≥4.65 | Download progress bars | [tqdm.github.io](https://tqdm.github.io) |
| 🧪 | **pytest** | ≥7.4 | Unit & integration testing | [docs.pytest.org](https://docs.pytest.org/en/stable/) |
| 🔍 | **Ruff** | ≥0.1 | Ultra-fast Python linter | [docs.astral.sh/ruff](https://docs.astral.sh/ruff/) |
| 🖤 | **Black** | ≥23.0 | Deterministic code formatter | [black.readthedocs.io](https://black.readthedocs.io/en/stable/) |
| �️ | **Pandera** | ≥0.20 | DataFrame validation schemas | [pandera.readthedocs.io](https://pandera.readthedocs.io/en/stable/) |
| �📊 | **matplotlib** | ≥3.8 | Training distribution charts for MLflow | [matplotlib.org](https://matplotlib.org/stable/) |
| ⚙️ | **PyYAML** | ≥6.0 | YAML configuration parsing | [pyyaml.org](https://pyyaml.org/wiki/PyYAMLDocumentation) |

---

## 📁 Project Structure

```
imdb-recommender/
│
├── 📂 api/                          # REST API layer
│   └── main.py                      # FastAPI app with Prometheus metrics
│
├── 📂 src/                          # Core ML source code
│   ├── 📂 ingestion/                # Data acquisition
│   │   ├── downloader.py            # IMDb dataset downloader
│   │   └── loader.py                # TSV parser & data merger
│   │
│   ├── 📂 features/                 # Feature engineering
│   │   └── engineering.py           # Filtering, encoding, scaling
│   │
│   ├── 📂 models/                   # ML models
│   │   ├── recommender.py           # ContentRecommender class
│   │   └── train.py                 # Full training pipeline
│   │
│   ├── 📂 inference/                # Serving layer
│   │   └── engine.py                # Production inference engine
│   │
│   ├── 📂 validation/               # Data validation
│   │   └── schemas.py               # Pandera schemas (5 schemas)
│   │
│   └── mlflow_tracking.py           # MLflow integration & stats
│
├── 📂 ui/                           # Frontend
│   └── app.py                       # Streamlit web application
│
├── 📂 configs/                      # Configuration
│   └── training_config.yaml         # Training pipeline config
│
├── 📂 monitoring/                   # Observability stack
│   ├── 📂 prometheus/
│   │   └── prometheus.yml           # Prometheus scrape config
│   └── 📂 grafana/
│       ├── 📂 provisioning/
│       │   ├── 📂 datasources/
│       │   │   └── prometheus.yml   # Grafana ← Prometheus wiring
│       │   └── 📂 dashboards/
│       │       └── dashboards.yml   # Dashboard auto-provisioner
│       └── 📂 dashboards/
│           └── recommender-overview.json  # 20+ panel dashboard
│
├── 📂 tests/                        # Test suite (56 tests)
│   ├── test_ingestion.py            # Data download/load tests
│   ├── test_features.py             # Feature engineering tests
│   ├── test_model.py                # Recommender model tests
│   ├── test_api.py                  # API endpoint tests
│   └── test_validation.py           # Pandera schema tests
│
├── 📂 .github/workflows/            # CI/CD
│   └── ci.yml                       # GitHub Actions pipeline
│
├── 📂 data/                         # Data directory (git-ignored)
│   ├── 📂 raw/                      # Downloaded IMDb TSVs
│   └── 📂 processed/                # Trained artifacts
│       ├── recommender_model.pkl    # Serialized model (~11 MB)
│       ├── title_metadata.parquet   # Title metadata for API
│       └── pipeline_manifest.json   # Training run manifest
│
├── Dockerfile                       # Container image definition
├── docker-compose.yml               # 5-service orchestration
├── Makefile                         # Build automation (20+ targets)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git exclusion rules
├── .dockerignore                    # Docker build exclusions
├── LICENSE                          # MIT License
└── README.md                        # You are here! 👋
```

### 🔑 Key Files Explained

<details>
<summary><strong>📥 <code>src/ingestion/downloader.py</code></strong> — IMDb Dataset Downloader</summary>

Downloads the official IMDb Non-Commercial datasets from [`datasets.imdbws.com`](https://datasets.imdbws.com). Features:

- **Idempotent downloads** — skips already-cached files unless `force=True`
- **Streaming with progress bars** — uses [`tqdm`](https://tqdm.github.io) for visual progress
- **Checksum verification** — MD5 hashes stored alongside downloads for integrity
- **Automatic decompression** — `.tsv.gz` → `.tsv` with cleanup

**Datasets downloaded:**
| File | Size | Contents |
|------|:----:|----------|
| `title.basics.tsv` | ~850 MB | All IMDb titles (10M+ rows): type, title, year, runtime, genres |
| `title.ratings.tsv` | ~27 MB | Ratings & vote counts for rated titles |
| `title.principals.tsv` | ~2.5 GB | Cast/crew (optional, disabled by default) |

</details>

<details>
<summary><strong>📂 <code>src/ingestion/loader.py</code></strong> — Data Loader & Merger</summary>

Loads raw TSV files into [pandas](https://pandas.pydata.org/docs/) DataFrames with proper type handling:

- Handles IMDb's `\N` null marker via `na_values` parameter
- Converts `startYear`, `runtimeMinutes` to numeric with `errors="coerce"`
- Merges `title.basics` + `title.ratings` on `tconst` (IMDb's unique title ID)
- Memory-efficient loading with explicit `dtype` declarations

</details>

<details>
<summary><strong>⚙️ <code>src/features/engineering.py</code></strong> — Feature Engineering Pipeline</summary>

Transforms raw data into a model-ready feature matrix through 5 steps:

| Step | Function | What It Does |
|:----:|----------|-------------|
| 1 | `filter_titles()` | Filters by title type, min votes (500), min year (1970), excludes adult content |
| 2 | `encode_genres()` | One-hot encodes the comma-separated `genres` column using [`MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) → 26 genre features |
| 3 | `scale_numerical_features()` | Scales `averageRating`, `numVotes`, `runtimeMinutes`, `startYear` to [0,1] using [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) → 4 numerical features |
| 4 | `create_text_features()` | Builds combined text string for future semantic embeddings |
| 5 | `build_feature_matrix()` | Assembles final `float32` numpy array of shape `(n_titles, 30)` |

**Result:** 79,596 titles × 30 features (4 numerical + 26 genre one-hot)

</details>

<details>
<summary><strong>🧠 <code>src/models/recommender.py</code></strong> — Content Recommender Model</summary>

The core ML model class `ContentRecommender` that powers recommendations:

- **`fit(feature_matrix, tconst_ids, titles)`** — Stores the feature matrix and builds case-insensitive title lookup indices
- **`recommend(title, top_k=10)`** — Computes [`cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) between the query title's feature vector and all other titles, returns top-k most similar
- **`search_titles(query, limit=20)`** — Partial string matching across all indexed titles
- **`find_title_index(title)`** — Case-insensitive fuzzy lookup (exact → prefix → substring)
- **`save(path)` / `load(path)`** — Pickle serialization with `HIGHEST_PROTOCOL` for efficiency

**How similarity works:**
```python
# For a given title, compute its similarity to ALL other titles:
query_vector = feature_matrix[query_idx : query_idx + 1]      # (1, 30)
similarities = cosine_similarity(query_vector, feature_matrix)  # (1, 79596)
top_k_indices = np.argsort(similarities)[::-1][1 : top_k + 1]  # exclude self
```

</details>

<details>
<summary><strong>🏋️ <code>src/models/train.py</code></strong> — Training Pipeline</summary>

The main training orchestrator that runs the full 6-step pipeline with Pandera validation gates:

```
Step 1/6: 📥 Download datasets       → Idempotent IMDb download (~1.1 GB)
Step 2/6: 📂 Load & merge            → Combine basics + ratings into one DataFrame
   └─ 2b: 🛡️ Pandera validation     → Validate merged data (MergedDatasetSchema)
Step 3/6: ⚙️  Feature engineering     → Filter → Encode → Scale → Build matrix
   └─ 3b: 🛡️ Pandera validation     → Validate features (FeatureEngineeredSchema)
Step 4/6: 🔧 Fit model               → ContentRecommender.fit() with 30 features
Step 5/6: 💾 Save artifacts           → .pkl model + .parquet metadata + .json manifest
Step 6/6: ✅ Validation               → Test queries to verify model works
```

Integrated with MLflow for full training observability. Computes and logs:
- Data distribution statistics (ratings, votes, years, runtime, genres)
- Feature analysis (per-feature stats, correlations, matrix properties)
- Similarity distribution (sampled cosine similarity across 500 titles)

</details>

<details>
<summary><strong>🔌 <code>src/inference/engine.py</code></strong> — Production Inference Engine</summary>

The `RecommendationEngine` class provides a high-level interface for the API layer:

- **`load()`** — Loads trained model (`.pkl`) and title metadata (`.parquet`) from disk
- **`recommend(title, top_k)`** — Returns enriched recommendations with full metadata (year, rating, votes, genres, runtime, type)
- **`search(query, limit)`** — Returns enriched search results with metadata
- **`get_available_titles_count()`** — Returns number of indexed titles

**Enrichment:** Raw model output only contains `tconst` + `similarity_score`. The engine joins against the metadata parquet to add year, rating, genres, etc.

</details>

<details>
<summary><strong>🚀 <code>api/main.py</code></strong> — FastAPI Recommendation Service</summary>

Production-grade REST API (~490 lines) with comprehensive instrumentation and MLflow inference tracing:

**Endpoints:**
| Method | Path | Description |
|:------:|------|-------------|
| `POST` | `/recommend` | Get top-k similar titles for a given movie/series |
| `GET` | `/search?q=...` | Search titles by partial name match |
| `GET` | `/health` | Health check with uptime, request count, avg latency |
| `GET` | `/metrics` | Prometheus metrics endpoint (15+ custom metrics) |
| `GET` | `/` | API info and navigation links |
| `GET` | `/docs` | Auto-generated Swagger UI (by FastAPI) |

**Middleware Features:**
- Request tracing with unique `X-Request-ID` headers
- Latency measurement in `X-Latency-Ms` headers
- Structured logging with timestamps and client IP
- CORS enabled for frontend integration

**Prometheus Metrics (15+ custom):**
- `recommender_requests_total` — Total requests by method/endpoint/status
- `recommender_request_latency_seconds` — Latency histograms (p50/p90/p99)
- `recommender_recommendations_total` — Recommendations by status (success/not_found/error)
- `recommender_recommendation_latency_seconds` — Recommendation computation time
- `recommender_similarity_score` — Distribution of returned similarity scores
- `recommender_searches_total` — Search requests count
- `recommender_model_titles_loaded` — Number of loaded titles (gauge)
- `recommender_model_started_at_seconds` — Startup timestamp for uptime calculation
- `recommender_errors_total` — Error count by endpoint and type
- And more...

**MLflow Inference Tracing:**
- `@mlflow.trace` decorators on `/recommend` and `/search` endpoints
- Span-level tracking with `mlflow.start_span` for engine calls
- Full input/output capture for every API request

</details>

<details>
<summary><strong>🎨 <code>ui/app.py</code></strong> — Streamlit Web Interface</summary>

A polished, ~610-line [Streamlit](https://docs.streamlit.io) app with a dark theme and IMDb-inspired design:

- **Search autocomplete** — Start typing and see matching titles instantly
- **Movie cards** — Rich cards with rank badges, similarity scores, genre tags
- **IMDb-gold accent** — `#f5c518` color scheme inspired by IMDb branding
- **Responsive layout** — `st.columns()` for side-by-side movie cards
- **Custom CSS** — 200+ lines of custom styling with Inter font, gradients, shadows
- **Error handling** — Graceful degradation when API is unavailable

</details>

<details>
<summary><strong>�️ <code>src/validation/schemas.py</code></strong> — Pandera Data Validation Schemas</summary>

Defines 5 [Pandera](https://pandera.readthedocs.io/) `DataFrameSchema` objects that validate data at every pipeline stage:

| Schema | Stage | Key Checks |
|--------|-------|------------|
| `TitleBasicsSchema` | Raw ingestion | `tconst` starts with "tt", `isAdult` ∈ {0,1}, `runtimeMinutes` > 0 |
| `TitleRatingsSchema` | Raw ingestion | `averageRating` ∈ [0, 10], `numVotes` > 0 |
| `MergedDatasetSchema` | After join | Unique `tconst`, all basics + ratings columns present |
| `FilteredDatasetSchema` | After filtering | Valid `titleType`, `isAdult == 0`, `startYear ≥ 1970`, non-null genres |
| `FeatureEngineeredSchema` | Model-ready | Scaled values ∈ [0, 1], at least one `genre_` column, dataset > 100 rows |

Includes `validate_dataframe()` helper with **lazy validation** — collects all errors before raising, so you see every issue in one pass.

</details>

<details>
<summary><strong>�📈 <code>src/mlflow_tracking.py</code></strong> — MLflow v3.10 Integration</summary>

Centralized MLflow v3.10 configuration (~780 lines) for maximum training and inference observability:

| Function | What It Logs |
|----------|-------------|
| `init_mlflow()` | Sets tracking URI, creates experiment, enables system metrics logging (CPU/RAM) |
| `log_training_run()` | 12-step comprehensive logging: params, metrics, dataset, stats, features, model metadata, tables, charts, model registry, text summaries, file artifacts, tags |
| `_log_sample_tables()` | 6 `mlflow.log_table` calls: top_50_rated, top_50_voted, genre_distribution, title_type_distribution, sample_recommendations, decade_distribution |
| `_log_distribution_figures()` | 8 matplotlib charts via `mlflow.log_figure`: rating distribution, votes (log), decades, genres (bar), runtime, rating vs votes scatter, feature correlation heatmap, similarity distribution |
| `_log_model_to_registry()` | Registers model as pyfunc with signature via `mlflow.pyfunc.log_model` and `infer_signature` |
| `_log_text_summary()` | Human-readable training summary via `mlflow.log_text` |
| `init_inference_tracing()` | Enables `mlflow.tracing` for real-time API inference tracing |

**Total per training run:** 60+ metrics, 8 charts, 6 tables, 6 file artifacts, 1 registered model

</details>

<details>
<summary><strong>📊 <code>monitoring/grafana/dashboards/recommender-overview.json</code></strong> — Grafana Dashboard</summary>

A pre-built, auto-provisioned Grafana dashboard with **20+ panels** across **6 sections**:

| Section | Panels |
|---------|--------|
| 🔑 Key Metrics | Request rate, P50/P99 latency, Error rate, Model titles, Model status, **Uptime** |
| 📈 Request Traffic | Requests by endpoint, Requests by status code |
| ⏱️ Latency Analysis | Latency percentiles (all endpoints), Recommendation-specific latency |
| 🎯 Recommendation Engine | Status breakdown (success/not_found/error), Similarity distribution, Top-K distribution |
| 🔍 Search Engine | Search rate, Results count distribution |
| ❌ Errors & Reliability | Errors by type, Error percentage, In-progress requests, Load time |

Auto-provisioned on startup via Grafana's file-based provisioning — no manual setup needed.

</details>

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

> **Prerequisites:** [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.

```bash
# 1. Clone the repository
git clone https://github.com/CheikhAiLabs/imbd-recommander.git
cd imbd-recommander

# 2. Train the model first (downloads ~1.1 GB of IMDb data)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make train

# 3. Start the full stack (5 services)
make docker-up
```

**🎉 That's it! Open these URLs:**

| Service | URL | Credentials |
|---------|-----|-------------|
| 🎨 **Web UI** | [localhost:9877](http://localhost:9877) | — |
| 🚀 **API Docs** | [localhost:9876/docs](http://localhost:9876/docs) | — |
| 📈 **MLflow** | [localhost:9878](http://localhost:9878) | — |
| 📊 **Grafana** | [localhost:9880](http://localhost:9880) | `admin` / `recommender` |
| 🔥 **Prometheus** | [localhost:9879](http://localhost:9879) | — |

### Option B — Local Development

```bash
# 1. Clone and set up virtual environment
git clone https://github.com/CheikhAiLabs/imbd-recommander.git
cd imbd-recommander
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
make install
# or: pip install -r requirements.txt

# 3. Train the model (downloads IMDb data + trains)
make train

# 4. Start services (in separate terminals)
make api       # Terminal 1: FastAPI on port 9876
make ui        # Terminal 2: Streamlit on port 9877
make mlflow    # Terminal 3: MLflow on port 9878
```

---

## 🧠 How It Works

### 1️⃣ Data Ingestion

The system downloads two official IMDb datasets from [`datasets.imdbws.com`](https://datasets.imdbws.com):

| Dataset | File | Rows | Key Columns |
|---------|------|:----:|-------------|
| **Title Basics** | `title.basics.tsv` | ~10.5M | `tconst`, `titleType`, `primaryTitle`, `startYear`, `runtimeMinutes`, `genres` |
| **Title Ratings** | `title.ratings.tsv` | ~1.4M | `tconst`, `averageRating`, `numVotes` |

These are merged on `tconst` (IMDb's unique identifier, e.g., `tt0111161` = *The Shawshank Redemption*).

> 📌 **Data license:** IMDb Non-Commercial Datasets are provided for personal and non-commercial use. See [IMDb Conditions of Use](https://www.imdb.com/conditions).

### 2️⃣ Feature Engineering

Raw data is transformed into a **30-dimensional feature vector** per title:

```
┌──────────────────────────────────────────────────────────────┐
│                    Feature Vector (30-D)                      │
├──────────────────────┬───────────────────────────────────────┤
│ Numerical (4)        │ Genre One-Hot (26)                    │
│ ┌──────────────────┐ │ ┌─────────────────────────────────┐   │
│ │ averageRating    │ │ │ genre_Action                    │   │
│ │ numVotes (log)   │ │ │ genre_Adventure                 │   │
│ │ runtimeMinutes   │ │ │ genre_Animation                 │   │
│ │ startYear        │ │ │ genre_Comedy                    │   │
│ │                  │ │ │ genre_Crime                     │   │
│ │ (MinMaxScaled    │ │ │ genre_Documentary               │   │
│ │  to [0, 1])      │ │ │ genre_Drama                     │   │
│ └──────────────────┘ │ │ ... (26 total genres)           │   │
│                      │ │ (binary: 0 or 1)                │   │
│                      │ └─────────────────────────────────┘   │
└──────────────────────┴───────────────────────────────────────┘
```

**Filtering criteria (configurable in `configs/training_config.yaml`):**
- `min_votes: 500` — At least 500 IMDb votes
- `min_year: 1970` — Released 1970 or later
- `title_types: [movie, tvSeries, tvMiniSeries, tvMovie]`
- Excludes adult content
- Requires non-null genres and runtime

### 3️⃣ Model Training

The `ContentRecommender` uses [**cosine similarity**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) to find the most similar titles:

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Where $A$ and $B$ are 30-dimensional feature vectors. The similarity score ranges from 0 (completely different) to 1 (identical features).

**At query time:**
1. Look up the query title's feature vector
2. Compute cosine similarity against all 79,596 other vectors
3. Sort by similarity descending
4. Return top-k results (excluding the query title itself)

### 4️⃣ Inference Engine

The `RecommendationEngine` wraps the trained model for production serving:

```
User Request                     API Response
    │                                ▲
    ▼                                │
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ FastAPI   │───▶│ Inference    │───▶│ Enriched     │
│ /recommend│    │ Engine       │    │ Response     │
│           │    │              │    │              │
│ title:    │    │ 1. Model     │    │ title, year  │
│ "Inception"   │    recommend()│    │ rating, votes│
│ top_k: 10 │    │              │    │ genres,      │
│           │    │ 2. Enrich w/ │    │ runtime,     │
│           │    │    metadata  │    │ similarity   │
└──────────┘    └──────────────┘    └──────────────┘
```

---

## 🎯 Training Pipeline

### Running Training

```bash
# Using Make (recommended)
make train

# Using Python directly
python -m src.models.train --config configs/training_config.yaml

# With custom config
python -m src.models.train --config path/to/my_config.yaml
```

**Expected output:**
```
══════════════════════════════════════════════════════════════════
IMDb RECOMMENDATION SYSTEM - TRAINING PIPELINE
══════════════════════════════════════════════════════════════════

📥 Step 1/6: Downloading datasets...
   Cached: data/raw/title.basics.tsv already exists. Skipping.
   Cached: data/raw/title.ratings.tsv already exists. Skipping.

📂 Step 2/6: Loading and merging datasets...
   Raw data: 1,418,647 titles
   🛡️ Step 2b: Validating merged data (MergedDatasetSchema)...
   ✅ MergedDatasetSchema — all checks passed

⚙️  Step 3/6: Feature engineering...
   Filtered titles: 1,418,647 -> 79,596 (5.6%)
   Encoded 26 genres: ['Action', 'Adventure', 'Animation', ...]
   Scaled averageRating: range [1.0, 10.0] -> [0, 1]
   Feature matrix shape: (79596, 30)
   🛡️ Step 3b: Validating features (FeatureEngineeredSchema)...
   ✅ FeatureEngineeredSchema — all checks passed

🔧 Step 4/6: Fitting recommendation model...
   Recommender fitted: 79,596 titles, 30 features

💾 Step 5/6: Saving artifacts...
   Model saved to data/processed/recommender_model.pkl (11.2 MB)

✅ Step 6/6: Validation...
   "Inception" → Top 3: Tenet (0.9847), Interstellar (0.9712), ...
   Pipeline completed in 42.3s
```

### What Gets Logged to MLflow

When the MLflow tracking server is running, every training run logs:

| Category | What's Logged | Count |
|----------|--------------|:-----:|
| **Parameters** | Full flattened config (data paths, filter thresholds, model name, etc.) | ~15 |
| **Metrics** | n_titles, n_features, n_genres, training_time_seconds, etc. | ~10 |
| **Data Stats** | Rating/vote/year/runtime distributions (mean, median, std, quartiles, skew) | ~20 |
| **Feature Analysis** | Per-feature statistics, matrix density, correlations | ~10 |
| **Similarity Stats** | Cosine similarity distribution across sampled titles | ~8 |
| **Dataset** | Full training DataFrame registered via `mlflow.data.from_pandas` with schema & profile | 1 |
| **Tables** | top_50_rated, top_50_voted, genre_distribution, title_type_distribution, sample_recommendations, decade_distribution | 6 |
| **Charts** | Rating distribution, votes (log), decades, genres (bar), runtime, rating vs votes scatter, feature correlation heatmap, similarity distribution | 8 |
| **Model Registry** | Registered pyfunc model (`imdb-content-recommender`) with inferred signature | 1 |
| **Text Summary** | Human-readable training report via `mlflow.log_text` | 1 |
| **File Artifacts** | `recommender_model.pkl`, `title_metadata.parquet`, `pipeline_manifest.json`, `data_statistics.json`, `feature_analysis.json`, `training.log` | 6 |
| **System Metrics** | CPU usage, RAM consumption during training (via `mlflow.enable_system_metrics_logging`) | auto |

**Total: 60+ metrics, 8 charts, 6 tables, 6 file artifacts, 1 registered model per run**

**Inference tracing:** Every `/recommend` and `/search` API call is traced via `@mlflow.trace` with full input/output span capture.

### Training Output Artifacts

| File | Size | Format | Description |
|------|:----:|:------:|-------------|
| `recommender_model.pkl` | ~11 MB | Pickle | Serialized model with feature matrix, title index, metadata |
| `title_metadata.parquet` | ~3 MB | Parquet | Title metadata for API enrichment (year, rating, genres, etc.) |
| `pipeline_manifest.json` | ~2 KB | JSON | Run metadata: timestamp, config, feature names, elapsed time |
| `training.log` | ~5 KB | Text | Full structured training log |

---

## 📡 API Reference

The API is built with [FastAPI](https://fastapi.tiangolo.com) and automatically generates interactive documentation at `/docs` ([Swagger UI](https://swagger.io/tools/swagger-ui/)) and `/redoc` ([ReDoc](https://redocly.com/redoc)).

### `POST /recommend`

Get top-k recommendations for a movie or series.

**Request:**
```bash
# Basic recommendation
curl -X POST http://localhost:9876/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception", "top_k": 5}'

# With tconst for disambiguation (e.g., multiple titles named "Parasite")
curl -X POST http://localhost:9876/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "Parasite", "tconst": "tt6751668", "top_k": 5}'
```

**Response:**
```json
{
  "query": {
    "tconst": "tt1375666",
    "title": "Inception",
    "year": 2010,
    "rating": 8.8,
    "votes": 2500000,
    "genres": "Action,Adventure,Sci-Fi",
    "runtime_minutes": 148
  },
  "top_k": 5,
  "recommendations": [
    {
      "rank": 1,
      "tconst": "tt6723592",
      "title": "Tenet",
      "similarity_score": 0.9847,
      "year": 2020,
      "rating": 7.3,
      "votes": 530000,
      "genres": "Action,Sci-Fi,Thriller",
      "runtime_minutes": 150,
      "title_type": "movie"
    }
  ],
  "latency_ms": 23.45
}
```

### `GET /search`

Search for titles by partial name match.

**Request:**
```bash
curl "http://localhost:9876/search?q=dark+knight&limit=5"
```

**Response:**
```json
[
  {
    "tconst": "tt0468569",
    "title": "The Dark Knight",
    "year": 2008,
    "rating": 9.0,
    "genres": "Action,Crime,Drama"
  },
  {
    "tconst": "tt1345836",
    "title": "The Dark Knight Rises",
    "year": 2012,
    "rating": 8.4,
    "genres": "Action,Thriller"
  }
]
```

### `GET /health`

Health check with operational metrics.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "titles_count": 79596,
  "version": "1.0.0",
  "total_requests": 142,
  "avg_latency_ms": 18.73,
  "uptime_seconds": 3600.5
}
```

### `GET /metrics`

Prometheus-format metrics for scraping. Returns 15+ custom metrics including request counts, latency histograms, recommendation statistics, error rates, and model status gauges. See the [Monitoring](#-monitoring--observability) section for details.

---

## 🖥️ Web Interface

The Streamlit UI provides a beautiful, dark-themed interface for exploring recommendations:

**Key Features:**
- 🔍 **Search autocomplete** — Start typing and see matching titles instantly
- 🃏 **Movie cards** — Rich cards with rank badges, similarity scores, genre tags
- ⭐ **Ratings & votes** — IMDb rating and vote count on every card
- 🎭 **Genre tags** — Colorful genre pills on each recommendation
- 📱 **Responsive layout** — Adapts to different screen sizes
- 🌙 **Dark theme** — IMDb-gold (`#f5c518`) accent color on dark background

**Access:** [http://localhost:9877](http://localhost:9877) (Docker) or `make ui` (local)

---

## 📊 Monitoring & Observability

The system provides **three layers** of observability:

### Prometheus Metrics

The API exposes a `/metrics` endpoint scraped by [Prometheus](https://prometheus.io/docs/introduction/overview/) every 5 seconds.

**15+ custom metrics:**

| Metric | Type | Description |
|--------|:----:|-------------|
| `recommender_requests_total` | Counter | Total requests (labels: method, endpoint, status_code) |
| `recommender_request_latency_seconds` | Histogram | Request latency with p50/p90/p99 percentiles |
| `recommender_requests_in_progress` | Gauge | Currently processing requests |
| `recommender_recommendations_total` | Counter | Recommendation requests (labels: status) |
| `recommender_recommendation_latency_seconds` | Histogram | Recommendation computation time |
| `recommender_top_k_requested` | Histogram | Distribution of requested top_k values |
| `recommender_similarity_score` | Histogram | Returned similarity score distribution |
| `recommender_searches_total` | Counter | Search requests by status |
| `recommender_search_results_count` | Histogram | Number of results per search |
| `recommender_model_titles_loaded` | Gauge | Number of titles in the model |
| `recommender_model_loaded` | Gauge | Model load status (1=loaded, 0=not) |
| `recommender_model_started_at_seconds` | Gauge | Startup timestamp (for uptime: `time() - value`) |
| `recommender_model_load_time_seconds` | Gauge | Startup model load time |
| `recommender_errors_total` | Counter | Errors by endpoint and type |
| `recommender_app_info` | Info | App version and metadata |

### Grafana Dashboard

A **20+ panel** dashboard auto-provisioned on startup. Access at [localhost:9880](http://localhost:9880) with `admin` / `recommender`.

| Dashboard Section | What You See |
|:-----------------:|-------------|
| 🔑 **Key Metrics** | Request rate (req/s), P50 & P99 latency, error rate %, model status, titles count |
| 📈 **Request Traffic** | Time-series of requests by endpoint, status code distribution over time |
| ⏱️ **Latency Analysis** | P50/P90/P99 latency percentiles for all endpoints and recommendation-specific |
| 🎯 **Recommendation Engine** | Success/failure breakdown, similarity score distribution, top_k histogram |
| 🔍 **Search Engine** | Search rate, results count distribution |
| ❌ **Errors & Reliability** | Error counts by type, error percentage, in-flight requests, model load time |

### MLflow Experiment Tracking

[MLflow v3.10](https://mlflow.org/docs/latest/index.html) provides deep experiment tracking for training and real-time API inference tracing. Access at [localhost:9878](http://localhost:9878).

**What you can explore:**
- 📊 **Compare training runs** — Side-by-side metrics comparison (60+ metrics per run)
- 📈 **Distribution charts** — 8 matplotlib charts: rating, votes, decades, genres, runtime, correlations, similarity
- 📋 **Data tables** — 6 tables: top-rated/voted titles, genre & type distributions, sample recommendations
- 🏷️ **Model Registry** — Registered `imdb-content-recommender` pyfunc model with inferred signature
- 📦 **Dataset tracking** — Full training dataset logged via `mlflow.data.from_pandas` with schema & profile
- 🔍 **Inference tracing** — Every `/recommend` and `/search` API call traced with inputs, outputs, and latency
- ✅ **GenAI quality scoring** — Evaluate responses with MLflow scorers (e.g., `Correctness`) and track quality metrics
- 💻 **System metrics** — CPU/RAM usage during training via `mlflow.enable_system_metrics_logging`
- 📁 **Artifacts** — Download model files, metadata, training logs from any run
- 🏷️ **Parameters** — Full training configuration captured as searchable params

**Run scorer evaluation:**
```bash
# Uses a mock predict function (no external API call)
make eval-genai

# Uses OpenAI as predict function (requires OPENAI_API_KEY)
PYTHONPATH=. python scripts/evaluate_genai.py \
  --tracking-uri http://localhost:9878 \
  --experiment-id 1 \
  --provider openai \
  --openai-model gpt-4o-mini
```
Default scorer set includes built-in `Correctness()` and a custom metric `expected_response_match/mean`.

---

## ⚙️ Configuration

All training behavior is controlled via [`configs/training_config.yaml`](configs/training_config.yaml):

```yaml
data:
  raw_dir: "data/raw"              # Where to download IMDb TSVs
  processed_dir: "data/processed"  # Where to save trained artifacts
  force_download: false            # Re-download even if cached?
  include_principals: false        # Include cast/crew data? (slower)

features:
  min_votes: 500                   # Minimum IMDb votes to include a title
  min_year: 1970                   # Earliest release year
  title_types:                     # Which title types to include
    - movie
    - tvSeries
    - tvMiniSeries
    - tvMovie

model:
  name: "content_recommender_v1"   # Model identifier

mlflow:
  enabled: true                    # Enable MLflow tracking?
  tracking_uri: "http://localhost:5555"  # MLflow server URL
  experiment_name: "imdb-recommender"   # Experiment name
```

**Tuning tips:**
- Lower `min_votes` → more titles, but more noise
- Higher `min_votes` → fewer titles, but higher quality recommendations
- Add `"tvSpecial"` to `title_types` to include TV specials
- Set `include_principals: true` to download cast/crew data (adds ~2.5 GB)

---

## 🐳 Docker Deployment

### Services & Ports

The `docker-compose.yml` defines **5 services** on collision-free ports:

| Service | Container | Internal Port | External Port | Image |
|---------|-----------|:-------------:|:-------------:|-------|
| 🚀 **API** | `imdb-api` | 8000 | **9876** | Custom (Dockerfile) |
| 🎨 **UI** | `imdb-ui` | 8501 | **9877** | Custom (Dockerfile) |
| 📈 **MLflow** | `imdb-mlflow` | 5000 | **9878** | `ghcr.io/mlflow/mlflow:v3.10.0` |
| 🔥 **Prometheus** | `imdb-prometheus` | 9090 | **9879** | `prom/prometheus:v3.2.1` |
| 📊 **Grafana** | `imdb-grafana` | 3000 | **9880** | `grafana/grafana:11.5.2` |

### Docker Commands

```bash
# Build images
make docker-build
# or: docker compose build

# Start all services (detached)
make docker-up
# or: docker compose up -d

# View logs
make docker-logs
# or: docker compose logs -f api

# Stop everything
make docker-down
# or: docker compose down

# Full cleanup (remove volumes + images)
make clean-docker
# or: docker compose down -v --rmi local
```

### Volumes

| Volume | Purpose |
|--------|---------|
| `mlflow-data` | MLflow database and artifacts (persists across restarts) |
| `prometheus-data` | Prometheus time-series data (30-day retention) |
| `grafana-data` | Grafana configuration and state |
| `./data` (bind mount) | Trained model artifacts shared with API container |

---

## 🧪 Testing

```bash
# Run all tests
make test
# or: python -m pytest tests/ -v --tb=short

# Run specific test file
python -m pytest tests/test_model.py -v

# Run with coverage (install pytest-cov first)
python -m pytest tests/ --cov=src --cov-report=term-missing
```

**Test suite (56 tests):**

| File | Count | What It Covers |
|------|:-----:|----------------|
| `tests/test_ingestion.py` | 6 | Dataset key validation, idempotent download skip, MD5 checksums, TSV parsing, null (`\N`) handling |
| `tests/test_features.py` | 10 | Title filtering (type/votes/year), genre one-hot encoding (binary, specific genre), numerical scaling ([0,1] range), feature matrix shape & NaN-free, text features |
| `tests/test_model.py` | 11 | Fit state, recommend count/self-exclusion/sort order/required keys, unknown title error, search, save/load round-trip, unfitted error, case-insensitive lookup, missing file error |
| `tests/test_api.py` | 8 | `GET /health` (200), `POST /recommend` (200/422/404), `GET /search` (200/422), `GET /` root info |
| `tests/test_validation.py` | 19 | Pandera schema enforcement: valid/invalid data for all 5 schemas, `validate_dataframe()` helper success & failure paths |

---

## 🛠️ Development

### Available Make Targets

```bash
make help           # Show all available targets

# ── Setup ──
make install        # Install Python dependencies

# ── Data ──
make download       # Download IMDb datasets (idempotent)
make train          # Full training pipeline (download + train)

# ── Serving ──
make api            # Start FastAPI on port 9876
make ui             # Start Streamlit on port 9877
make mlflow         # Start MLflow on port 9878
make eval-genai     # Run MLflow GenAI evaluation (Correctness scorer)

# ── Docker ──
make docker-build   # Build Docker images
make docker-up      # Start 5-service stack
make docker-down    # Stop all services
make docker-logs    # Tail service logs

# ── Quality ──
make test           # Run pytest suite
make lint           # Run Ruff linter
make format         # Format with Black + Ruff

# ── Maintenance ──
make clean          # Remove generated files
make clean-all      # Remove everything including data
make clean-docker   # Remove Docker volumes & images

# ── Full Pipeline ──
make all            # install → train → test
```

### Code Quality

This project uses two industry-standard tools:

- **[Ruff](https://docs.astral.sh/ruff/)** — Ultra-fast Python linter (replaces flake8, isort, pyupgrade, and more)
- **[Black](https://black.readthedocs.io/en/stable/)** — The uncompromising Python code formatter

```bash
# Lint check
make lint

# Auto-format
make format
```

---

## 🗺️ Roadmap

- [ ] 🔮 **Hybrid recommender** — Combine content-based + collaborative filtering
- [ ] 🤖 **Sentence embeddings** — Use transformer models for semantic title similarity
- [ ] 👥 **User profiles** — Personalized recommendations based on watch history
- [ ] 🔄 **Auto-retrain** — Scheduled retraining when new IMDb data is available
- [ ] 📊 **A/B testing** — Compare recommendation strategies in production
- [ ] 🌐 **API rate limiting** — Token bucket rate limiter for production traffic
- [ ] 📱 **React frontend** — Modern SPA alternative to Streamlit

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

**Development setup:**
```bash
git clone https://github.com/CheikhAiLabs/imbd-recommander.git
cd imbd-recommander
python3 -m venv .venv && source .venv/bin/activate
make install
make train
make test  # verify everything works
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> **IMDb Data:** The datasets used in this project are provided by IMDb for personal and non-commercial use under the [IMDb Non-Commercial Licensing](https://developer.imdb.com/non-commercial-datasets/). The data files themselves are **not** included in this repository — they are downloaded at training time directly from IMDb's servers.

---

## 🙏 Acknowledgments

- **[IMDb](https://www.imdb.com)** — For providing the [Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)
- **[scikit-learn](https://scikit-learn.org)** — For `cosine_similarity`, `MinMaxScaler`, and `MultiLabelBinarizer`
- **[FastAPI](https://fastapi.tiangolo.com)** — For the high-performance, auto-documented REST framework
- **[MLflow](https://mlflow.org)** — For experiment tracking and artifact management
- **[Prometheus](https://prometheus.io)** + **[Grafana](https://grafana.com)** — For production-grade monitoring
- **[Streamlit](https://streamlit.io)** — For the rapid UI prototyping framework

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/CheikhAiLabs">CheikhAiLabs</a></sub>
</p>

<p align="center">
  <a href="#-imdb-recommender-system">⬆ Back to Top</a>
</p>
