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
from src.explainability.lime_explainer import LimeExplainer
from src.explainability.visualizer import generate_text_heatmap, generate_outcome_summary

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
        
        # Lazy load explainer
        self._explainer = None
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

    def predict(self, code: str, threshold: float = 0.5, calibrate: bool = False) -> dict:
        """Predict whether a code snippet is vulnerable.

        Args:
            code: C/C++ source code string.
            threshold: Confidence threshold for flagging as vulnerable.
            calibrate: If True, apply logit adjustment to compensate for 
                       class imbalance in training data (DiverseVul is ~94% safe).

        Returns:
            dict: ``{"is_vulnerable": bool, "confidence": float}``.
        """
        device = self.current_device
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits.squeeze(0)
        
        # NOTE: Manual logit calibration (+0.65) was removed because the model 
        # is now trained on a 1:1 balanced dataset. Scaling is no longer needed.
        
        probs = torch.softmax(logits, dim=-1)
        vuln_prob = probs[1].item()
        is_vulnerable = vuln_prob > threshold

        return {
            "is_vulnerable": is_vulnerable,
            "confidence": float(vuln_prob if is_vulnerable else 1 - vuln_prob),
            "vuln_probability": float(vuln_prob),
            "raw_logits": logits.tolist()
        }

    def explain(self, code: str) -> list[dict]:
        """Generate local explanations using LIME."""
        if self._explainer is None:
            self._explainer = LimeExplainer(self.model, self.tokenizer)
        return self._explainer.explain(code)


# ── CLI Entry Point ──────────────────────────────────────────────────────


def main() -> None:
    """CLI utility for single-snippet predictions."""
    import argparse
    from rich.console import Console
    from rich.panel import Panel
    from rich.status import Status
    from rich.table import Table

    console = Console()
    parser = argparse.ArgumentParser(description="Predict vulnerability in C/C++ code.")
    parser.add_argument("--code", type=str, required=True, help="Code snippet to analyze")
    parser.add_argument("--explain", action="store_true", help="Generate explainability heatmap")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (0.0-1.0)")
    parser.add_argument("--calibrate", action="store_true", help="Enable logit adjustment calibration (not needed for 1:1 balanced models)")
    args = parser.parse_args()

    # Initialise predictor (loads model)
    with Status("[bold cyan]Loading model...", console=console):
        try:
            predictor = VulnerabilityPredictor.get_instance()
        except FileNotFoundError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return

    # Run prediction
    with Status("[bold cyan]Analyzing code...", console=console):
        result = predictor.predict(args.code, threshold=args.threshold, calibrate=args.calibrate)
    
    is_vuln = result["is_vulnerable"]
    confidence = result["confidence"]
    status_color = "red" if is_vuln else "green"
    status_icon = "🚨 VULNERABLE" if is_vuln else "✅ SAFE"
    
    panel_content = (
        f"Result: [bold {status_color}]{status_icon}[/bold {status_color}]\n"
        f"Confidence: [bold]{confidence * 100:.2f}%[/bold]\n\n"
        "[dim]Note: This is an AI prediction. Always use human verification.[/dim]"
    )
    
    console.print("\n")
    console.print(Panel(
        panel_content,
        title="[bold white]Vulnerability Analysis Result[/bold white]",
        border_style=status_color,
        padding=(1, 2),
        expand=False
    ))

    # Optional Explainability
    if args.explain:
        with Status("[bold yellow]Generating heatmaps (LIME)...", console=console):
            explanation = predictor.explain(args.code)
            heatmap = generate_text_heatmap(explanation)
            summary, findings = generate_outcome_summary(is_vuln, confidence, explanation)

        console.print("\n[bold white]AI TRANSPARENCY REPORT[/bold white]")
        console.print("═" * 30)
        
        # Findings Table
        table = Table(title="Key Indicators", box=None, show_header=False)
        table.add_column("Finding", style="dim")
        for f in findings:
            table.add_row(f"• {f}")
        console.print(table)

        # Heatmap
        console.print(Panel(
            heatmap,
            title="[bold yellow]Token Impact Heatmap[/bold yellow]",
            border_style="yellow",
            padding=(1, 1),
            expand=False
        ))
        
        console.print("\n[dim]Model: microsoft/codebert-base | Framework: LIME[/dim]")


if __name__ == "__main__":
    main()
