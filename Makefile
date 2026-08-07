.PHONY: install lint test run-dashboard docker-build

install:
	pip install -e .

lint:
	python3 -m ruff check src tests || true

test:
	pytest tests/ --cov=src

run-dashboard:
	python3 dashboards/dash_app.py

docker-build:
	docker build -t renewable-energy-forecasting:latest .
