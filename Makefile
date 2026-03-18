.PHONY: docs docs-serve docs-deploy test lint format clean

# Documentation
docs:
	cp notebooks/*.ipynb docs/notebooks/
	mkdocs build

docs-serve:
	cp notebooks/*.ipynb docs/notebooks/
	mkdocs serve

docs-deploy:
	cp notebooks/*.ipynb docs/notebooks/
	mkdocs gh-deploy --force

# Development
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

clean:
	rm -rf site/ docs/notebooks/*.ipynb
