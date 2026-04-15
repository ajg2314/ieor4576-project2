.PHONY: dev

dev:
	uv run uvicorn app:app --reload --port 8080 --reload-exclude 'artifacts'
