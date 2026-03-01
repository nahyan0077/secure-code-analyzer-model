"""
Inference module for single code-snippet predictions.

Provides a ``VulnerabilityPredictor`` class that loads the model once
and exposes a ``predict()`` method returning a structured result.
"""

from pathlib import Path
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from configs.config import Config
from src.model.model_loader import load_trained_model
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VulnerabilityPredictor:
    """Singleton-ready predictor for vulnerability detection.

    Loads the fine-tuned model and tokenizer once, then provides
    fast inference via :meth:`predict`.
    """

    _instance: Optional["VulnerabilityPredictor"] = None

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_path: Path | str = Config.MODEL_DIR,
    ) -> None:
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model, self.tokenizer = load_trained_model(model_path)

        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()
        logger.info("VulnerabilityPredictor initialised on %s", self.device)

    @classmethod
    def get_instance(cls, **kwargs) -> "VulnerabilityPredictor":
        """Return or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @property
    def current_device(self) -> torch.device:
        """Get the actual device the model is currently on."""
        return next(self.model.parameters()).device

    def predict(self, code: str) -> dict:
        """Predict whether a code snippet is vulnerable.

        Args:
            code: C/C++ source code string.

        Returns:
            dict: ``{"is_vulnerable": bool, "confidence": float}``.
        """
        device = self.current_device
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**encoding)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        vuln_prob = probs[1].item()
        is_vulnerable = vuln_prob > 0.5

        return {
            "is_vulnerable": is_vulnerable,
            "confidence": round(vuln_prob if is_vulnerable else 1 - vuln_prob, 4),
        }
