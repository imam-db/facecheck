PYTHON ?= python3

.PHONY: install models test lint eval serve docker-build

install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/pip install -e ".[dev]"

models:
	.venv/bin/python scripts/download_models.py

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check src tests eval scripts

eval:
	.venv/bin/python scripts/fetch_lfw.py
	.venv/bin/python -m eval.run_lfw

serve:
	.venv/bin/uvicorn facecheck.api:app --reload --port 8000

docker-build:
	docker build -t facecheck:dev .
