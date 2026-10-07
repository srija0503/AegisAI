# ══════════════════════════════════════════════════════════════
#  Makefile — Quantum-Secure Federated Firewall
#  Usage: make <target>
# ══════════════════════════════════════════════════════════════

.PHONY: help install setup data train simulate test lint docker-up docker-down clean

help:
	@echo ""
	@echo "  Quantum-Secure Federated Firewall — Makefile Targets"
	@echo "  ────────────────────────────────────────────────────"
	@echo "  install        Install Python dependencies"
	@echo "  setup          Full first-time setup (install + data + preprocess)"
	@echo "  data           Download all datasets"
	@echo "  preprocess     Preprocess raw datasets"
	@echo "  split          Run stratified train/val/test split"
	@echo "  train          Train initial RL model"
	@echo "  simulate       Run multi-node simulation"
	@echo "  test           Run all tests with coverage"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-int       Run integration tests only"
	@echo "  lint           Run ruff linter"
	@echo "  docker-up      Start all services via Docker Compose"
	@echo "  docker-down    Stop all Docker services"
	@echo "  clean          Remove generated files"
	@echo ""

# ── Environment ────────────────────────────────────────────────
install:
	pip install -r requirements.txt

setup: install data preprocess split
	@echo "✅ Setup complete. Run 'make train' to train the RL agent."

# ── Data pipeline ──────────────────────────────────────────────
data:
	python datasets/download_dataset.py --dataset all

preprocess:
	python datasets/preprocess.py --dataset nsl-kdd
	python datasets/preprocess.py --dataset cicids2017

split:
	python datasets/train_test_split.py --dataset nsl-kdd

# ── Training ───────────────────────────────────────────────────
train:
	python scripts/train_initial_model.py

# ── Simulation ─────────────────────────────────────────────────
simulate:
	python scripts/start_simulation.py

node-01:
	python scripts/start_node.py --node-id node_01

node-02:
	python scripts/start_node.py --node-id node_02

node-03:
	python scripts/start_node.py --node-id node_03

server:
	python scripts/start_server.py

# ── Testing ────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-int:
	pytest tests/integration/ -v

test-security:
	pytest tests/security/ -v

# ── Linting ────────────────────────────────────────────────────
lint:
	ruff check . --fix

format:
	ruff format .

# ── Docker ─────────────────────────────────────────────────────
docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ── Cleanup ────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
	find . -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned."
