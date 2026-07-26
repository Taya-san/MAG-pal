.PHONY: venv install test test-all clean

venv:
	python3 -m venv .venv
	@echo "Run: source .venv/bin/activate"

install:
	pip install -r requirements.txt

test:
	python3 tests/run_all.py

test-all:
	python3 tests/run_all.py

clean:
	rm -rf .venv __pycache__ */__pycache__ *.pyc memory.db
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
