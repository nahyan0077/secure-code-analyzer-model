"""
Training pipeline for the vulnerability detection model.

Uses HuggingFace Trainer with class-weighted loss to handle the
heavily imbalanced DiverseVul dataset on Apple Silicon MPS.

Usage::

    python -m src.model.train
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback

# Ensure project root is on path when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import Config
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import (
    VulnerabilityDataset,
    clean_data,
    compute_class_weights,
    split_data,
)
from src.model.model_loader import load_model, load_tokenizer, save_model
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Weighted Trainer ───────────────────────────────────────────────────────────


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to CrossEntropyLoss for imbalanced data."""

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
    """Compute accuracy, precision, recall, F1, and ROC AUC for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)  # softmax
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        metrics["roc_auc"] = 0.0  # single-class batch edge case
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full training pipeline."""
    Config.ensure_dirs()
    device = get_device()

    # ── Pipeline Header ──────────────────────────────────────────────────
    
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    logger.info("[bold white]🚀 STARTING VULNERABILITY DETECTION TRAINING PIPELINE[/bold white]")
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    # 1. Load Data (Memory Optimized Two-Pass)
    logger.info("[bold yellow]Phase 1:[/bold yellow] Data Acquisition & Curation (Streaming Optimization)")
    
    # Pass 1: Load metadata ONLY
    from src.data.dataset_loader import load_dataset_indices, load_selected_records
    df_meta = load_dataset_indices(Config.DATA_PATH)
    
    # Apply MAX_SAMPLES limit if set (for local training on limited hardware)
    if Config.MAX_SAMPLES and len(df_meta) > Config.MAX_SAMPLES:
        logger.info("Subsampling to [bold yellow]%d[/bold yellow] records (MAX_SAMPLES)", Config.MAX_SAMPLES)
        df_meta = df_meta.sample(n=Config.MAX_SAMPLES, random_state=42).reset_index(drop=True)
    
    # Split the indices (using metadata only)
    train_indices_df, val_indices_df, test_indices_df = split_data(df_meta)
    
    # NOTE: NO undersampling — we train on the full natural distribution
    # and use class-weighted loss instead to handle the imbalance.
    # This lets the model learn the real-world prior (~94% safe / 6% vuln).
    
    # Limit validation indices for local performance
    if Config.MAX_VAL_SAMPLES and len(val_indices_df) > Config.MAX_VAL_SAMPLES:
        logger.info("Subsampling val indices to %d for performance", Config.MAX_VAL_SAMPLES)
        val_indices_df = val_indices_df.sample(n=Config.MAX_VAL_SAMPLES, random_state=42)
        
    # Pass 2: Selective Load code only for needed indices
    logger.info("Selective loading code for final subsets...")
    train_df = load_selected_records(Config.DATA_PATH, set(train_indices_df["file_index"]))
    val_df = load_selected_records(Config.DATA_PATH, set(val_indices_df["file_index"]))
    
    # Cleanup metadata structures
    del df_meta, train_indices_df, val_indices_df, test_indices_df
    gc.collect()

    # ── Class Distribution Report ────────────────────────────────────────
    for name, subset in [("Train", train_df), ("Val", val_df)]:
        vuln = (subset["target"] == 1).sum()
        safe = (subset["target"] == 0).sum()
        total = len(subset)
        logger.info(
            "  %s set: [bold green]%d[/bold green] samples "
            "([dim]%d vuln %.1f%% | %d safe %.1f%%[/dim])",
            name, total, vuln, 100 * vuln / total, safe, 100 * safe / total,
        )

    # 2. Tokenizer & datasets
    logger.info("[bold yellow]Phase 2:[/bold yellow] Tokenization & Dataset Preparation")
    
    # Apply Balanced Undersampling for Precision Boost (Recommended for BERT)
    from src.data.preprocessing import balance_data
    train_df = balance_data(train_df, strategy="undersample")
    
    tokenizer = load_tokenizer()
    train_dataset = VulnerabilityDataset(train_df, tokenizer=tokenizer)
    val_dataset = VulnerabilityDataset(val_df, tokenizer=tokenizer)
    
    # 3. Class weights — Disabbled for Balanced Training
    logger.info("Injecting [bold yellow]stronger MLP classifier head[/bold yellow] for non-BERT-like base model")
    # Using neutral weights (1.0) because data is now physically balanced 50/50
    class_weights = torch.ones(Config.NUM_LABELS)
    logger.info("Using balanced data strategy (neutral weights: %s)", class_weights.tolist())

    # Eagerly release dataframes now that Dataset objects hold the data
    logger.info("Releasing [dim]train_df/val_df[/dim] memory ...")
    del train_df
    del val_df
    gc.collect()

    logger.info("Successfully created [dim]train (%d)[/dim] and [dim]val (%d)[/dim] datasets.", 
                len(train_dataset), len(val_dataset))


    # 4. Model
    logger.info("[bold yellow]Phase 4:[/bold yellow] Model Initialization")
    model = load_model()
    model.to(device)
    logger.info("Model loaded and mapped to %s", device)

    # 5. Training arguments
    params = Config.TRAIN_PARAMS.copy()
    
    output_dir = Config.MODEL_DIR / "checkpoints"
    total_steps = (len(train_dataset) // params["batch_size"]) * params["epochs"]
    warmup_steps = int(total_steps * params.get("warmup_ratio", 0.1))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
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
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=device.type == "cuda",
        dataloader_num_workers=2 if device.type == "cuda" else 0,
    )

    # 6. Train with WeightedTrainer (class-weighted loss for imbalance)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=params["early_stopping_patience"])]
    )

    trainer.train()

    # 6b. Save training history (loss curves for overfitting analysis)
    history = trainer.state.log_history
    epoch_metrics = []
    for entry in history:
        if "eval_loss" in entry:
            epoch_metrics.append({
                "epoch": entry.get("epoch"),
                "eval_loss": round(entry["eval_loss"], 6),
                "eval_accuracy": round(entry.get("eval_accuracy", 0), 4),
                "eval_f1": round(entry.get("eval_f1", 0), 4),
                "eval_precision": round(entry.get("eval_precision", 0), 4),
                "eval_recall": round(entry.get("eval_recall", 0), 4),
                "eval_roc_auc": round(entry.get("eval_roc_auc", 0), 4),
            })
    # Attach training loss from the last log entry before each eval
    train_losses = [e for e in history if "loss" in e and "eval_loss" not in e]
    for i, em in enumerate(epoch_metrics):
        # Find the closest training loss entry for this epoch
        matching = [t for t in train_losses if t.get("epoch", 0) <= em["epoch"]]
        if matching:
            em["train_loss"] = round(matching[-1]["loss"], 6)

    history_path = Config.REPORTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(epoch_metrics, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    # Log summary table
    logger.info("[bold cyan]Per-Epoch Summary:[/bold cyan]")
    for em in epoch_metrics:
        logger.info(
            "  Epoch %.0f — train_loss=%.4f  eval_loss=%.4f  eval_f1=%.4f  eval_roc_auc=%.4f",
            em.get("epoch", 0), em.get("train_loss", 0),
            em["eval_loss"], em["eval_f1"], em["eval_roc_auc"],
        )

    # 7. Save
    logger.info("[bold yellow]Phase 6:[/bold yellow] Export & Persistence")
    save_model(model, tokenizer, Config.MODEL_DIR)
    
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    logger.info("[bold green]✅ PIPELINE COMPLETE - Model saved to %s[/bold green]", Config.MODEL_DIR)
    logger.info("[bold cyan]" + "=" * 60 + "[/bold cyan]")


if __name__ == "__main__":
    main()
