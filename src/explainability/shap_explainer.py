"""
SHAP-based explainability for the vulnerability detection model.

Generates per-token contribution scores that indicate which tokens
in a code snippet most influenced the vulnerability prediction.
"""

from typing import Optional

import numpy as np
import shap
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from configs.config import Config
from src.model.model_loader import load_trained_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """Model-agnostic explainer using SHAP for the vulnerability classifier.
    Uses ``shap.Explainer`` with a custom prediction wrapper.
    Inference for SHAP is forced to CPU for compatibility.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_path: str | None = None,
    ) -> None:
        """
        Args:
            model: Pre-loaded model (optional).
            tokenizer: Pre-loaded tokenizer (optional).
            model_path: Path to saved model. Used if model/tokenizer not given.
        """
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            path = model_path or str(Config.MODEL_DIR)
            self.model, self.tokenizer = load_trained_model(path)

        self.model.eval()

        self._explainer = shap.Explainer(
            self._predict_proba,
            self.tokenizer,
            output_names=["safe", "vulnerable"],
        )
        logger.info("ShapExplainer initialised")

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """Prediction function compatible with SHAP Explainer.

        Args:
            texts: List of code strings.

        Returns:
            np.ndarray: Probability array of shape ``(n, 2)``.
        """
        device = self._device
        results = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=Config.MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**encoding)

            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
            results.append(probs)

        return np.array(results)

    def explain(self, code: str, max_tokens: int = 50) -> list[dict]:
        """Generate per-token SHAP contribution scores.

        Args:
            code: C/C++ code snippet.
            max_tokens: Maximum number of top contributing tokens to return.

        Returns:
            List of ``{"token": str, "score": float}`` dicts, sorted by
            absolute contribution (descending). Positive scores push toward
            *vulnerable*, negative toward *safe*.
        """
        logger.info("Generating SHAP explanation (max_tokens=%d)", max_tokens)

        try:
            shap_values = self._explainer([code])

            # shap_values.values shape: (1, num_tokens, num_classes)
            # We want the "vulnerable" class column (index 1)
            token_values = shap_values.values[0]  # (num_tokens, 2)
            vuln_scores = token_values[:, 1]  # scores for "vulnerable"
            tokens = shap_values.data[0]  # token strings

            # Pair tokens with scores, sort by |score| descending
            token_scores = [
                {"token": str(tok), "score": round(float(score), 6)}
                for tok, score in zip(tokens, vuln_scores)
                if str(tok).strip()  # skip empty tokens
            ]
            token_scores.sort(key=lambda x: abs(x["score"]), reverse=True)

            return token_scores[:max_tokens]

        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            return [{"token": "error", "score": 0.0, "detail": str(e)}]
