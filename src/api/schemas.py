"""
Pydantic request / response schemas for the FastAPI inference API.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for the ``/predict`` endpoint."""

    code: str = Field(
        ...,
        description="C/C++ source code snippet to analyse.",
        min_length=1,
    )
    include_explanation: bool = Field(
        default=False,
        description="If true, include explanation token scores (slower).",
    )
    explainer: str = Field(
        default="shap",
        description="Which explainer to use: 'shap' or 'lime'.",
        pattern="^(shap|lime)$"
    )
    threshold: float = Field(
        default=0.2,
        description="Confidence threshold to flag as vulnerable (default 0.2). Lower increases recall.",
        ge=0.0,
        le=1.0
    )
    calibrate: bool = Field(
        default=False,
        description="Apply prior-scaling to compensate for class imbalance (not needed for new 1:1 model).",
    )


class TokenScore(BaseModel):
    """A single token's contribution score."""

    token: str
    score: float


class PredictionResponse(BaseModel):
    """Response body for the ``/predict`` endpoint."""

    is_vulnerable: bool = Field(
        ..., description="Whether the code is predicted to be vulnerable."
    )
    confidence: float = Field(
        ..., description="Confidence score for the prediction (0–1)."
    )
    vuln_probability: Optional[float] = Field(
        default=None, description="Raw probability of the 'vulnerable' class (0–1)."
    )
    raw_logits: Optional[list[float]] = Field(
        default=None, description="Raw logit values from the model before activation."
    )
    explanation: Optional[list[TokenScore]] = Field(
        default=None,
        description="Per-token contribution scores (only when requested).",
    )
    heatmap: Optional[str] = Field(
        default=None,
        description="Text-based ASCII heatmap of contributions.",
    )
    findings: Optional[list[str]] = Field(
        default=None,
        description="List of key tokens contributing to the prediction.",
    )
    outcome_summary: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the model decision.",
    )


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint."""

    status: str = "ok"
    model_loaded: bool = False
