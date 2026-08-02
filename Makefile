PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: help venv install test test-fast test-slow golden lint fmt typecheck docs docs-build clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the Python 3.12 virtualenv
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip

install: venv  ## Install the package in editable mode with dev+docs extras
	$(PY) -m pip install -e ".[dev,docs]"

test:  ## Run the full test suite
	$(PYTEST) -q

test-fast:  ## Run everything except the slow Monte-Carlo studies
	$(PYTEST) -q -m "not slow"

test-slow:  ## Run only the slow bootstrap / level studies
	$(PYTEST) -q -m slow

golden:  ## Regenerate golden fixtures from R (requires R + the copula package)
	bash tools/rgolden/run_all.sh

lint:  ## Lint with ruff
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:  ## Autoformat with ruff
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

typecheck:  ## Type-check with mypy
	.venv/bin/mypy rcopula

docs:  ## Serve the documentation locally
	.venv/bin/mkdocs serve

docs-build:  ## Build the documentation, failing on any broken reference
	.venv/bin/mkdocs build --strict

clean:  ## Remove build and cache artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
