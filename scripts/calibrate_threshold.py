import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import json
from pathlib import Path
import sys

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import Config
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import VulnerabilityDataset, clean_data, split_data
from src.model.model_loader import load_trained_model
from src.utils.device import get_device

def calibrate():
    device = get_device()
    print("Loading model for calibration...")
    model, tokenizer = load_trained_model(Config.MODEL_DIR)
    model.to(device)
    model.eval()

    print("Loading validation data...")
    df = load_dataset(Config.DATA_PATH, max_samples=Config.MAX_SAMPLES)
    df = clean_data(df)
    _, val_df, _ = split_data(df)
    
    val_dataset = VulnerabilityDataset(val_df, tokenizer=tokenizer)
    loader = DataLoader(val_dataset, batch_size=Config.TRAIN_PARAMS["batch_size"], collate_fn=DataCollatorWithPadding(tokenizer))

    print("Gathering raw probabilities...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("\nSearching for optimal threshold...")
    best_f1 = 0
    best_threshold = 0.5
    
    # Test thresholds from 0.05 to 0.99
    thresholds = np.linspace(0.05, 0.99, 100)
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Final metrics at best threshold
    final_preds = (all_probs >= best_threshold).astype(int)
    print(f"\nOptimal Results (Threshold: {best_threshold:.3f})")
    print(f"F1 Score:  {best_f1:.4f}")
    print(f"Precision: {precision_score(all_labels, final_preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, final_preds, zero_division=0):.4f}")
    print(f"Accuracy:  {accuracy_score(all_labels, final_preds):.4f}")

    return best_threshold

if __name__ == "__main__":
    calibrate()
