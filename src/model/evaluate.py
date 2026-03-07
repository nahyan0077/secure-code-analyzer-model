"""
Evaluation module for the vulnerability detection model.

Loads the test split, runs inference, and computes classification
metrics (accuracy, precision, recall, F1). Results are persisted to
``reports/metrics.json``.

Usage::

    python -m src.model.evaluate
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import Config
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import VulnerabilityDataset, clean_data, split_data
from src.model.model_loader import load_trained_model
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(threshold: float = 0.5) -> dict:
    """Run evaluation on the test set and save metrics.

    Args:
        threshold: Classification threshold (default 0.5).

    Returns:
        dict: Dictionary with accuracy, precision, recall, f1, and confusion matrix.
    """
    Config.ensure_dirs()
    device = get_device()

    # 1. Load data — use same split to get the test partition
    df = load_dataset(Config.DATA_PATH, max_samples=Config.MAX_SAMPLES)
    df = clean_data(df)
    _, _, test_df = split_data(df)

    # 2. Load model
    model, tokenizer = load_trained_model(Config.MODEL_DIR)
    model.to(device)
    model.eval()

    # 3. Dataset & loader
    test_dataset = VulnerabilityDataset(test_df, tokenizer=tokenizer)
    loader = DataLoader(
        test_dataset, 
        batch_size=Config.TRAIN_PARAMS["batch_size"],
        collate_fn=DataCollatorWithPadding(tokenizer)
    )

    # 4. Inference
    all_preds: list[int] = []
    all_probs: list[float] = []  # softmax P(vulnerable) for ROC AUC
    all_labels: list[int] = []

    logger.info("Running evaluation on %d samples …", len(test_dataset))
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Use custom threshold instead of simple argmax
            preds = (probs[:, 1] >= threshold).astype(int)

            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())  # P(vulnerable)
            all_labels.extend(labels.numpy().tolist())

    # 5. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 6. Metrics
    metrics = {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "f1": round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(all_labels, all_probs), 4),
        "total_samples": len(all_labels),
        "positive_samples": sum(all_labels),
        "negative_samples": len(all_labels) - sum(all_labels),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
    }

    # 7. Save
    metrics_path = Config.REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("Confusion Matrix: TP=%d, FP=%d, TN=%d, FN=%d", tp, fp, tn, fn)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics


if __name__ == "__main__":
    import os
    # Read threshold from environment variable, default to 0.5
    threshold = float(os.getenv("THRESHOLD", 0.5))
    if threshold != 0.5:
        logger.info("Using custom threshold: %.3f", threshold)
    evaluate(threshold=threshold)
