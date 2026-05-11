.PHONY: install test lint html serve clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=src/wyolo tests/

lint:
	pylint src/wyolo || true
	bandit -r src/wyolo

html:
	sphinx-build -b html docs/source/ docs/build/html

serve: html
	python3 -m http.server 8000 --directory docs/build/html

clean:
	rm -rf docs/build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
