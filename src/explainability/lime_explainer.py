"""
LIME (Local Interpretable Model-agnostic Explanations) for CodeBERT.

Provides an alternative local explanation by perturbing the input 
and training a linear model to approximate the local decision boundary.
"""

import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
from src.utils.logger import get_logger
from src.utils.device import get_device

logger = get_logger(__name__)

class LimeExplainer:
    def __init__(self, model, tokenizer):
        """Initialize LIME explainer.
        
        Args:
            model: Trained CodeBERT model.
            tokenizer: CodeBERT tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize LIME text explainer
        self.explainer = LimeTextExplainer(
            class_names=["Safe", "Vulnerable"],
            split_expression=r"(?<=[;{}])\s*",  # Split AFTER statements (semicolons/braces)
            random_state=42
        )
        logger.info("LimeExplainer initialised")

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _predict_probs(self, texts: list[str]) -> np.ndarray:
        """Prediction function for LIME.
        
        Args:
            texts: List of code snippets to predict.
            
        Returns:
            np.ndarray: Probabilities of shape (len(texts), 2).
        """
        device = self._device
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
        return probs

    def explain(self, code: str, num_features: int = 10) -> list[dict]:
        """Generate LIME explanation for a code snippet.
        
        Args:
            code: Source code string.
            num_features: Number of top features (tokens) to return.
            
        Returns:
            list[dict]: List of tokens and their importance scores.
        """
        exp = self.explainer.explain_instance(
            code,
            self._predict_probs,
            num_features=num_features,
            num_samples=500  # Number of perturbations
        )
        
        # Format the results
        explanation = []
        for token, score in exp.as_list():
            explanation.append({
                "token": token,
                "score": float(score)
            })
            
        return explanation
