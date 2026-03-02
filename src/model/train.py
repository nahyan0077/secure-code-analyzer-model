"""
Training pipeline for the vulnerability detection model.

Uses HuggingFace Trainer with class-weighted loss to handle the
heavily imbalanced DiverseVul dataset on Apple Silicon MPS.

Usage::

    python -m src.model.train
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments

# Ensure project root is on path when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import Config
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import (
    VulnerabilityDataset,
    balance_data,
    clean_data,
    compute_class_weights,
    split_data,
)
from src.model.model_loader import load_model, load_tokenizer, save_model
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Weighted Trainer ─────────────────────────────────────────────────────


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics callback ─────────────────────────────────────────────────────


def compute_metrics(eval_pred) -> dict:
    """Compute accuracy, precision, recall, and F1 for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full training pipeline."""
    Config.ensure_dirs()
    device = get_device()

    # ── Pipeline Header ──────────────────────────────────────────────────
    
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    logger.info("[bold white]🚀 STARTING VULNERABILITY DETECTION TRAINING PIPELINE[/bold white]")
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    # 1. Load Data
    logger.info("[bold yellow]Phase 1:[/bold yellow] Data Acquisition & Curation")
    df = load_dataset(Config.DATA_PATH, max_samples=Config.MAX_SAMPLES)
    df = clean_data(df)

    # Balance the dataset BEFORE splitting so both train and val are balanced
    logger.info("[bold yellow]Phase 1b:[/bold yellow] Balancing Dataset")
    df = balance_data(df)

    train_df, val_df, _ = split_data(df)

    # 2. Tokenizer & datasets
    logger.info("[bold yellow]Phase 2:[/bold yellow] Tokenization & Dataset Preparation")
    tokenizer = load_tokenizer()
    train_dataset = VulnerabilityDataset(train_df, tokenizer=tokenizer)
    val_dataset = VulnerabilityDataset(val_df, tokenizer=tokenizer)
    logger.info("Successfully created [dim]train (%d)[/dim] and [dim]val (%d)[/dim] datasets.", 
                len(train_dataset), len(val_dataset))

    # 3. Class weights
    logger.info("[bold yellow]Phase 3:[/bold yellow] Solving Class Imbalance")
    class_weights = compute_class_weights(train_df)

    # 4. Model
    logger.info("[bold yellow]Phase 4:[/bold yellow] Model Initialization")
    model = load_model()
    model.to(device)
    logger.info("CodeBERT loaded and mapped to [bold green]%s[/bold green]", device)

    # 5. Training arguments
    params = Config.TRAIN_PARAMS
    output_dir = Config.MODEL_DIR / "checkpoints"
    
    # Compute warmup steps from ratio
    total_steps = (len(train_dataset) // params["batch_size"]) * params["epochs"]
    warmup_steps = int(total_steps * params.get("warmup_ratio", 0.1))

    logger.info("Configuring TrainingArguments [dim](epochs=%d, batch=%d, lr=%s)[/dim]",
                params["epochs"], params["batch_size"], params["learning_rate"])
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy=params["eval_strategy"],
        save_strategy=params["save_strategy"],
        logging_steps=params["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=params["seed"],
        report_to="none",
    )

    # 6. Train
    logger.info("[bold yellow]Phase 5:[/bold yellow] Model Fine-tuning [dim](this may take a while)[/dim]")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 7. Save
    logger.info("[bold yellow]Phase 6:[/bold yellow] Export & Persistence")
    save_model(model, tokenizer, Config.MODEL_DIR)
    
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    logger.info("[bold green]✅ PIPELINE COMPLETE - Model saved to %s[/bold green]", Config.MODEL_DIR)
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")


if __name__ == "__main__":
    main()
