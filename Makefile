.PHONY: setup train evaluate dev clean help

# Default values
HOST ?= 0.0.0.0
PORT ?= 8000

help:
	@echo "Vulnerability Detector Development Commands"
	@echo "-------------------------------------------"
	@echo "setup    : Install dependencies using uv"
	@echo "train    : Train the model (all samples by default)"
	@echo "evaluate : Run model evaluation"
	@echo "dev      : Start the FastAPI inference service"
	@echo "clean    : Remove all temporary files and checkpoints"

setup:
	uv sync

train:
ifdef MAX_SAMPLES
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 MAX_SAMPLES=$(MAX_SAMPLES) uv run python -m src.model.train
else
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python -m src.model.train
endif

evaluate:
ifdef MAX_SAMPLES
	MAX_SAMPLES=$(MAX_SAMPLES) uv run python -m src.model.evaluate
else
	uv run python -m src.model.evaluate
endif

dev:
	uv run python -m uvicorn src.api.main:app --host $(HOST) --port $(PORT) --reload

clean:
	rm -rf model/checkpoints/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
