# ==============================================================================
# IMDb Recommender System - Makefile
# ==============================================================================
# Production-grade build automation for the recommendation pipeline.
#
# Usage:
#   make install     - Install Python dependencies
#   make download    - Download IMDb datasets
#   make train       - Run training pipeline (download + train)
#   make api         - Start the FastAPI server
#   make ui          - Start the Streamlit UI
#   make mlflow      - Start MLflow tracking server
#   make docker-up   - Start full Docker stack
#   make docker-down - Stop Docker stack
#   make test        - Run unit tests
#   make all         - Full pipeline: install, train, test
# ==============================================================================

.PHONY: install download train api ui mlflow test lint format clean all help
.PHONY: docker-up docker-down docker-build docker-logs

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON     := python3
PIP        := pip
CONFIG     := configs/training_config.yaml
API_HOST   := 0.0.0.0
API_PORT   := 9876
UI_PORT    := 9877
MLFLOW_PORT := 9878

# ── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo ""
	@echo "  🎬 IMDb Recommender System"
	@echo "  =========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
install: ## Install Python dependencies
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# ── Data Pipeline ────────────────────────────────────────────────────────────
download: ## Download IMDb datasets (idempotent)
	$(PYTHON) -c "from src.ingestion.downloader import download_all_datasets; download_all_datasets()"
	@echo "✅ Datasets downloaded"

# ── Training ─────────────────────────────────────────────────────────────────
train: ## Run complete training pipeline
	@mkdir -p data/processed
	$(PYTHON) -m src.models.train --config $(CONFIG)
	@echo "✅ Training complete"

# ── Serving ──────────────────────────────────────────────────────────────────
api: ## Start the FastAPI recommendation server (port $(API_PORT))
	uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --reload
	
ui: ## Start the Streamlit UI (port $(UI_PORT))
	streamlit run ui/app.py --server.port $(UI_PORT)

mlflow: ## Start MLflow tracking server (port $(MLFLOW_PORT))
	mlflow server \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root ./mlartifacts

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build: ## Build Docker images
	docker compose build
	@echo "✅ Docker images built"

docker-up: ## Start full stack (API + UI + MLflow + Prometheus + Grafana)
	docker compose up -d
	@echo ""
	@echo "🎬 ═══════════════════════════════════════════════════════"
	@echo "   IMDb Recommender Stack is UP!"
	@echo ""
	@echo "   🌐 API        → http://localhost:9876/docs"
	@echo "   🎨 UI         → http://localhost:9877"
	@echo "   📊 MLflow     → http://localhost:9878"
	@echo "   🔥 Prometheus → http://localhost:9879"
	@echo "   📈 Grafana    → http://localhost:9880  (admin/recommender)"
	@echo "🎬 ═══════════════════════════════════════════════════════"

docker-down: ## Stop all Docker services
	docker compose down
	@echo "✅ Stack stopped"

docker-logs: ## Tail logs from all services
	docker compose logs -f

# ── Quality ──────────────────────────────────────────────────────────────────
test: ## Run unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Run linter (ruff)
	ruff check src/ api/ tests/

format: ## Format code (black + ruff)
	black src/ api/ tests/ ui/
	ruff check --fix src/ api/ tests/

# ── Maintenance ──────────────────────────────────────────────────────────────
clean: ## Remove generated files and caches
	rm -rf data/raw/*.tsv
	rm -rf data/processed/*
	rm -rf __pycache__ **/__pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	@echo "✅ Cleaned"

clean-all: clean ## Remove everything including downloaded data
	rm -rf data/raw/*
	rm -rf mlruns/ mlartifacts/
	@echo "✅ All data removed"

clean-docker: ## Remove Docker volumes and images
	docker compose down -v --rmi local
	@echo "✅ Docker cleaned"

# ── Full Pipeline ────────────────────────────────────────────────────────────
all: install train test ## Full pipeline: install → train → test
	@echo ""
	@echo "🎬 ════════════════════════════════════════"
	@echo "   IMDb Recommender System is READY!"
	@echo "   Run 'make api' to start the API server"
	@echo "   Run 'make ui' to start the web interface"
	@echo "   Run 'make docker-up' for the full stack"
	@echo "🎬 ════════════════════════════════════════"
