.PHONY: venv install test test-all clean docker-build docker-run

# ----- VENV -----
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

# ----- DOCKER -----
docker-build:
	docker build -t mag-pal .

docker-run:
	docker run --rm -it \
		-v $(PWD)/.env:/app/.env:ro \
		-v mag-pal-data:/app/data \
		mag-pal

docker-test:
	docker build -t mag-pal .
	docker run --rm mag-pal python3 tests/run_all.py

docker-shell:
	docker run --rm -it --entrypoint /bin/bash mag-pal
