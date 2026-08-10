.PHONY: install dev-install lint test run-dashboard run-pipeline docker-build clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev-install:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

lint:
	$(RUFF) check src tests scripts

test:
	$(PYTEST) tests/

run-pipeline:
	$(PYTHON) scripts/run_pipeline.py

run-dashboard:
	$(PYTHON) dashboards/dash_app.py

docker-build:
	docker build -t renewable-energy-forecasting:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
