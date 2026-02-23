"""
FastAPI Recommendation Service
================================
Production-grade REST API for movie & series recommendations.
Includes Prometheus metrics, MLflow request tracing, structured logging,
and comprehensive health monitoring.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.tracing
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.inference.engine import RecommendationEngine

# ─── Structured Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    "recommender_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "recommender_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)
REQUEST_IN_PROGRESS = Gauge(
    "recommender_requests_in_progress",
    "Number of requests currently being processed",
    registry=REGISTRY,
)

# Recommendation-specific metrics
RECOMMENDATION_COUNT = Counter(
    "recommender_recommendations_total",
    "Total number of recommendation requests",
    ["status"],
    registry=REGISTRY,
)
RECOMMENDATION_LATENCY = Histogram(
    "recommender_recommendation_latency_seconds",
    "Recommendation computation latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=REGISTRY,
)
RECOMMENDATION_TOP_K = Histogram(
    "recommender_top_k_requested",
    "Distribution of requested top_k values",
    buckets=[3, 5, 10, 15, 20, 25, 30, 40, 50],
    registry=REGISTRY,
)
SIMILARITY_SCORES = Histogram(
    "recommender_similarity_score",
    "Distribution of similarity scores in recommendations",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    registry=REGISTRY,
)

# Search metrics
SEARCH_COUNT = Counter(
    "recommender_searches_total",
    "Total number of search requests",
    ["status"],
    registry=REGISTRY,
)
SEARCH_RESULTS = Histogram(
    "recommender_search_results_count",
    "Number of results returned per search",
    buckets=[0, 1, 5, 10, 15, 20, 50, 100],
    registry=REGISTRY,
)

# Model metrics
MODEL_TITLES_LOADED = Gauge(
    "recommender_model_titles_loaded",
    "Number of titles currently loaded in the model",
    registry=REGISTRY,
)
MODEL_LOADED = Gauge(
    "recommender_model_loaded",
    "Whether the model is currently loaded (1=yes, 0=no)",
    registry=REGISTRY,
)
MODEL_LOAD_TIME = Gauge(
    "recommender_model_load_time_seconds",
    "Time taken to load the model at startup",
    registry=REGISTRY,
)
MODEL_STARTED_AT = Gauge(
    "recommender_model_started_at_seconds",
    "Unix timestamp when the model was loaded (for uptime calculation)",
    registry=REGISTRY,
)

# Error tracking
ERROR_COUNT = Counter(
    "recommender_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
    registry=REGISTRY,
)

# App info
APP_INFO = Info(
    "recommender_app",
    "Application metadata",
    registry=REGISTRY,
)

# ─── Global State ──────────────────────────────────────────────────────────────
engine: RecommendationEngine | None = None
_request_count = 0
_total_latency_ms = 0.0
_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and initialize MLflow on startup."""
    global engine, _startup_time
    _startup_time = time.time()

    logger.info("Loading recommendation engine...")
    model_dir = Path("data/processed")

    load_start = time.time()
    engine = RecommendationEngine(model_dir=model_dir)
    engine.load()
    load_elapsed = time.time() - load_start

    titles_count = engine.get_available_titles_count()
    logger.info(f"Engine ready: {titles_count:,} titles loaded in {load_elapsed:.2f}s")

    # Update Prometheus gauges
    MODEL_LOADED.set(1)
    MODEL_TITLES_LOADED.set(titles_count)
    MODEL_LOAD_TIME.set(round(load_elapsed, 4))
    MODEL_STARTED_AT.set(_startup_time)
    APP_INFO.info(
        {
            "version": "1.0.0",
            "model_name": "content_recommender_v1",
            "titles_count": str(titles_count),
        }
    )

    # Initialize MLflow for inference tracing (v3.10 Tracing API)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:9878")
    try:
        from src.mlflow_tracking import init_inference_tracing

        init_inference_tracing(
            tracking_uri=tracking_uri,
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "imdb-recommender"),
        )
        logger.info(f"MLflow inference tracing enabled -> {tracking_uri}")
    except Exception as e:
        logger.warning(f"MLflow tracing init skipped: {e}")

    yield

    MODEL_LOADED.set(0)
    logger.info("Shutting down recommendation engine")


# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="IMDb Recommender API",
    description=(
        "Content-based Movie & Series Recommendation System "
        "powered by IMDb data. Find similar movies and series "
        "based on genres, ratings, and structured features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request Tracing Middleware ────────────────────────────────────────────────


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """Instrument every request with Prometheus metrics and structured logging."""
    global _request_count, _total_latency_ms

    # Skip metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    request_id = str(uuid.uuid4())[:8]
    endpoint = request.url.path
    method = request.method
    start = time.time()

    REQUEST_IN_PROGRESS.inc()

    logger.info(
        f"[{request_id}] {method} {endpoint} "
        f"(client={request.client.host if request.client else 'unknown'})"
    )

    response = await call_next(request)

    latency = time.time() - start
    latency_ms = latency * 1000
    status_code = str(response.status_code)

    REQUEST_IN_PROGRESS.dec()
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    _request_count += 1
    _total_latency_ms += latency_ms

    logger.info(f"[{request_id}] {status_code} completed in {latency_ms:.1f}ms")

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


# ─── Request/Response Models ──────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    """Request body for recommendation endpoint."""

    title: str = Field(
        ...,
        description="Movie or series title to get recommendations for",
        min_length=1,
        max_length=500,
        examples=["Inception"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to return",
    )
    tconst: str | None = Field(
        default=None,
        description="IMDb tconst ID for exact title identification (e.g. tt6751668)",
        examples=["tt6751668"],
    )


class TitleInfo(BaseModel):
    """Title metadata."""

    tconst: str | None = None
    title: str
    year: int | None = None
    rating: float | None = None
    votes: int | None = None
    genres: str | None = None
    runtime_minutes: int | None = None
    title_type: str | None = None


class RecommendationItem(BaseModel):
    """A single recommendation."""

    rank: int
    tconst: str
    title: str
    similarity_score: float
    year: int | None = None
    rating: float | None = None
    votes: int | None = None
    genres: str | None = None
    runtime_minutes: int | None = None
    title_type: str | None = None


class RecommendResponse(BaseModel):
    """Response body for recommendation endpoint."""

    query: TitleInfo
    top_k: int
    recommendations: list[RecommendationItem]
    latency_ms: float


class SearchResult(BaseModel):
    """A single search result."""

    tconst: str
    title: str
    year: int | None = None
    rating: float | None = None
    genres: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    titles_count: int
    version: str
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def prometheus_metrics():
    """Expose Prometheus metrics for scraping."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint with operational metrics."""
    avg_latency = (_total_latency_ms / _request_count) if _request_count > 0 else 0.0
    uptime = time.time() - _startup_time if _startup_time > 0 else 0.0

    return HealthResponse(
        status="healthy" if engine and engine._is_loaded else "degraded",
        model_loaded=engine._is_loaded if engine else False,
        titles_count=engine.get_available_titles_count() if engine and engine._is_loaded else 0,
        version="1.0.0",
        total_requests=_request_count,
        avg_latency_ms=round(avg_latency, 2),
        uptime_seconds=round(uptime, 2),
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
@mlflow.trace(name="POST /recommend", span_type="CHAIN")
async def recommend(request: RecommendRequest):
    """
    Get movie/series recommendations based on content similarity.

    Finds titles similar to the input based on genres, ratings,
    runtime, year, and other structured features.
    """
    if not engine or not engine._is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not loaded",
        )

    start = time.time()
    RECOMMENDATION_TOP_K.observe(request.top_k)

    try:
        with mlflow.start_span(name="engine.recommend") as span:
            span.set_inputs(
                {
                    "title": request.title,
                    "top_k": request.top_k,
                    "tconst": request.tconst or "",
                }
            )
            result = engine.recommend(
                title=request.title,
                top_k=request.top_k,
                tconst=request.tconst,
            )
            span.set_outputs(
                {
                    "query_title": result["query"].get("title", ""),
                    "n_results": len(result["recommendations"]),
                    "top_similarity": (
                        result["recommendations"][0]["similarity_score"]
                        if result["recommendations"]
                        else 0
                    ),
                }
            )
        RECOMMENDATION_COUNT.labels(status="success").inc()
    except ValueError as e:
        RECOMMENDATION_COUNT.labels(status="not_found").inc()
        ERROR_COUNT.labels(endpoint="/recommend", error_type="not_found").inc()
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Title not found",
                "message": str(e),
                "suggestion": "Use the /search endpoint to find available titles",
            },
        ) from None
    except Exception as e:
        RECOMMENDATION_COUNT.labels(status="error").inc()
        ERROR_COUNT.labels(endpoint="/recommend", error_type="internal").inc()
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during recommendation",
        ) from e

    latency = time.time() - start
    latency_ms = latency * 1000
    RECOMMENDATION_LATENCY.observe(latency)

    # Track similarity score distribution
    for rec in result["recommendations"]:
        SIMILARITY_SCORES.observe(rec["similarity_score"])

    return RecommendResponse(
        query=TitleInfo(**result["query"]),
        top_k=result["top_k"],
        recommendations=[RecommendationItem(**r) for r in result["recommendations"]],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/search", response_model=list[SearchResult], tags=["Search"])
@mlflow.trace(name="GET /search", span_type="RETRIEVER")
async def search_titles(
    q: str = Query(
        ...,
        min_length=1,
        max_length=200,
        description="Search query for movie/series titles",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results",
    ),
):
    """
    Search for movies and series by title.

    Performs a partial string match against all indexed titles.
    Use this to find the exact title before requesting recommendations.
    """
    if not engine or not engine._is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not loaded",
        )

    try:
        with mlflow.start_span(name="engine.search") as span:
            span.set_inputs({"query": q, "limit": limit})
            results = engine.search(query=q, limit=limit)
            span.set_outputs({"n_results": len(results)})
        SEARCH_COUNT.labels(status="success").inc()
        SEARCH_RESULTS.observe(len(results))
        return [SearchResult(**r) for r in results]
    except Exception as e:
        SEARCH_COUNT.labels(status="error").inc()
        ERROR_COUNT.labels(endpoint="/search", error_type="internal").inc()
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during search",
        ) from e


@app.get("/", tags=["System"])
async def root():
    """API root with basic info."""
    return {
        "name": "IMDb Recommender API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }
