"""
FastAPI application for vulnerability detection inference.

Loads the fine-tuned CodeBERT model at startup and exposes
a ``POST /predict`` endpoint for real-time predictions.

Usage::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.lime_explainer import LimeExplainer
from src.explainability.visualizer import generate_text_heatmap, generate_outcome_summary
from src.model.predict import VulnerabilityPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Globals (initialised at startup) ─────────────────────────────────────
_predictor: Optional[VulnerabilityPredictor] = None
_explainers: dict = {"shap": None, "lime": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and optional explainers at application startup."""
    global _predictor, _explainers

    logger.info("Loading vulnerability detection model …")
    try:
        _predictor = VulnerabilityPredictor.get_instance()
        logger.info("Model loaded successfully ✓")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise

    # Load explainers
    try:
        _explainers["shap"] = ShapExplainer()
        logger.info("SHAP explainer loaded ✓")
    except Exception as e:
        logger.warning("SHAP explainer failed: %s", e)

    try:
        _explainers["lime"] = LimeExplainer(_predictor.model, _predictor.tokenizer)
        logger.info("LIME explainer loaded ✓")
    except Exception as e:
        logger.warning("LIME explainer failed: %s", e)

    yield

    logger.info("Shutting down …")


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vulnerability Detection API",
    description="Detect security vulnerabilities in C/C++ code using an optimized BERT-base transformer. High-transparency edition.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict whether a code snippet is vulnerable.

    Supports dual explainability (SHAP/LIME) and automated transparency summaries.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = _predictor.predict(
            request.code, 
            threshold=request.threshold,
            calibrate=request.calibrate
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    explanation = None
    heatmap = None
    outcome_summary = None
    findings = None

    if request.include_explanation:
        explainer = _explainers.get(request.explainer)
        if explainer is None:
            logger.warning("%s explainer not available", request.explainer.upper())
        else:
            try:
                explanation = explainer.explain(request.code)
                heatmap = generate_text_heatmap(explanation)
                outcome_summary, findings = generate_outcome_summary(
                    result["is_vulnerable"], 
                    result["confidence"], 
                    explanation
                )
            except Exception as e:
                logger.error("%s explanation failed: %s", request.explainer.upper(), e)
                explanation = [{"token": "error", "score": 0.0}]

    return PredictionResponse(
        is_vulnerable=result["is_vulnerable"],
        confidence=result["confidence"],
        vuln_probability=result.get("vuln_probability"),
        raw_logits=result.get("raw_logits"),
        explanation=explanation,
        heatmap=heatmap,
        findings=findings,
        outcome_summary=outcome_summary
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
    )
