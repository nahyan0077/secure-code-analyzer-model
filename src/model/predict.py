"""
Inference module for single code-snippet predictions.

Provides a ``VulnerabilityPredictor`` class that loads the model once
and exposes a ``predict()`` method returning a structured result.
"""

from pathlib import Path
from typing import Optional
import os

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from configs.config import Config
from src.model.model_loader import load_trained_model
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.explainability.lime_explainer import LimeExplainer
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.visualizer import generate_text_heatmap, generate_outcome_summary
from src.utils.visualizer import plot_attention_heatmap
import subprocess
import numpy as np

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
        
        # Lazy load explainers
        self._explainer = None
        self._shap_explainer = None
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

    def predict(self, code: str, threshold: float = 0.5, calibrate: bool = False, output_attentions: bool = False) -> dict:
        """Predict whether a code snippet is vulnerable."""
        return self._predict_impl(code, threshold, calibrate, output_attentions)

    def _predict_impl(self, code: str, threshold: float = 0.5, calibrate: bool = False, output_attentions: bool = False) -> dict:
        device = self.current_device
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**encoding, output_attentions=output_attentions)

        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        vuln_prob = probs[1].item()
        is_vulnerable = vuln_prob > threshold

        result = {
            "is_vulnerable": is_vulnerable,
            "confidence": float(vuln_prob if is_vulnerable else 1 - vuln_prob),
            "vuln_probability": float(vuln_prob),
            "raw_logits": logits.tolist()
        }

        if output_attentions and hasattr(outputs, "attentions") and outputs.attentions:
            # attentions is a tuple of (layer_1, layer_2, ..., layer_N)
            # each layer has shape (batch, heads, seq, seq)
            # We'll take the LAST layer, first batch item
            last_layer_attentions = outputs.attentions[-1][0].cpu().numpy()
            result["attentions"] = last_layer_attentions
            result["tokens"] = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        return result

    def explain(self, code: str) -> list[dict]:
        """Generate local explanations using LIME."""
        if self._explainer is None:
            self._explainer = LimeExplainer(self.model, self.tokenizer)
        return self._explainer.explain(code)

    def explain_shap(self, code: str, output_plot_path: Optional[str] = None) -> list[dict]:
        """Generate local explanations using SHAP."""
        if self._shap_explainer is None:
            self._shap_explainer = ShapExplainer(self.model, self.tokenizer)
        return self._shap_explainer.explain(code, output_plot_path=output_plot_path)


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
    parser.add_argument("--explain", action="store_true", help="Generate explainability heatmap (LIME)")
    parser.add_argument("--visualize", action="store_true", help="Generate and open graphical plots (SHAP & Confusion Matrix)")
    parser.add_argument("--attention", action="store_true", help="Generate and open attention weight heatmap")
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
        result = predictor.predict(
            args.code, 
            threshold=args.threshold, 
            calibrate=args.calibrate, 
            output_attentions=args.attention
        )
    
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

    plots_to_open = []

    # Optional Attention Heatmap
    if args.attention and "attentions" in result:
        with Status("[bold blue]Generating attention visual analysis...", console=console):
            attn_plot_path = "reports/plots/latest_prediction_attention.png"
            os.makedirs(os.path.dirname(attn_plot_path), exist_ok=True)
            plot_attention_heatmap(result["attentions"], result["tokens"], attn_plot_path)
            plots_to_open.append(attn_plot_path)

    # Optional Visualization (Graphical - SHAP & CM)
    if args.visualize:
        # 1. SHAP Token Importance
        with Status("[bold magenta]Generating SHAP visual analysis...", console=console):
            shap_plot_path = "reports/plots/latest_prediction_shap.png"
            os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
            predictor.explain_shap(args.code, output_plot_path=shap_plot_path)
            plots_to_open.append(shap_plot_path)
            
        # 2. Confusion Matrix Heatmap (if exists)
        cm_plot_path = "reports/plots/confusion_matrix_heatmap.png"
        if os.path.exists(cm_plot_path):
            plots_to_open.append(cm_plot_path)
            
    if plots_to_open:
        console.print(f"\n[bold magenta]Visualization Mode Enabled:[/bold magenta] Opening {len(plots_to_open)} graphs...")
        
        import platform
        for plot in plots_to_open:
            try:
                system = platform.system()
                if system == "Darwin":  # macOS
                    subprocess.run(["open", plot], check=False)
                elif system == "Windows":
                    os.startfile(os.path.abspath(plot))
                else:  # Linux and others
                    subprocess.run(["xdg-open", plot], check=False)
            except Exception as e:
                console.print(f"[dim red]Could not open plot {plot}: {e}[/dim red]")

    # Optional Explainability (Text-based)
    if args.explain:
        with Status("[bold yellow]Generating heatmaps (LIME)...", console=console):
            explanation = predictor.explain(args.code)
            heatmap = generate_text_heatmap(explanation)
            # summary, findings = generate_outcome_summary(is_vuln, confidence, explanation) # generate_outcome_summary is missing in current file?

        console.print("\n[bold white]AI TRANSPARENCY REPORT (LIME)[/bold white]")
        console.print("═" * 30)
        
        # Heatmap
        console.print(Panel(
            heatmap,
            title="[bold yellow]Token Impact Heatmap[/bold yellow]",
            border_style="yellow",
            padding=(1, 1),
            expand=False
        ))
        
        console.print("\n[dim]Model: microsoft/codebert-base | Framework: LUME[/dim]")


if __name__ == "__main__":
    main()
