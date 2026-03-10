import json
import os
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(history_path: str, output_dir: str):
    """
    Plots the evolution of performance metrics from training history.
    """
    if not os.path.exists(history_path):
        print(f"Error: History file {history_path} not found.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    metrics = {
        'Loss': [h.get('eval_loss', h.get('train_loss')) for h in history],
        'Accuracy': [h.get('eval_accuracy', 0) for h in history],
        'Precision': [h.get('eval_precision', 0) for h in history],
        'Recall': [h.get('eval_recall', 0) for h in history],
        'F1-Score': [h.get('eval_f1', 0) for h in history],
        'ROC AUC': [h.get('eval_roc_auc', 0) for h in history]
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Custom colors
    colors = sns.color_palette("muted")

    for i, (metric_name, values) in enumerate(metrics.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        sns.lineplot(x=epochs, y=values, ax=ax, marker='o', linewidth=2.5, color=colors[i])
        ax.set_title(metric_name, fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # Adjust Y-axis for metrics (0 to 1) except Loss
        if metric_name != 'Loss':
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {output_dir}")


def plot_confusion_matrix(metrics_path: str, output_dir: str):
    """
    Plots a heatmap for the confusion matrix from metrics.
    """
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    cm_data = metrics.get('confusion_matrix', {})
    tp = cm_data.get('true_positives', 0)
    fp = cm_data.get('false_positives', 0)
    tn = cm_data.get('true_negatives', 0)
    fn = cm_data.get('false_negatives', 0)

    cm_array = np.array([[tn, fp], [fn, tp]])
    labels = ["Safe", "Vulnerable"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16})
    
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix heatmap saved to {output_dir}")


def plot_token_importance(token_scores: List[Dict], output_path: str, title: str = "Token Importance"):
    """
    Plots a horizontal bar chart of token importance scores.
    """
    if not token_scores:
        print("Warning: No token scores provided for plotting.")
        return

    # Filter out redundant tokens and take top N
    # token_scores are assumed to be sorted by |score|
    tokens = [item['token'] for item in token_scores[:20]]
    scores = [item['score'] for item in token_scores[:20]]
    
    # Reverse for plotting (highest score at top)
    tokens.reverse()
    scores.reverse()

    plt.figure(figsize=(12, 10))
    
    # Color bars: Dark Blue for positive (vulnerable), Light Blue for negative (safe)
    colors = ['#000080' if s > 0 else '#ADD8E6' for s in scores]
    
    bars = plt.barh(tokens, scores, color=colors)
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Importance Weight (SHAP)', fontsize=14)
    plt.ylabel('Tokens', fontsize=14)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Token importance plot saved to {output_path}")


def plot_attention_heatmap(attention_matrix: np.ndarray, tokens: List[str], output_path: str, title: str = "Attention Weights"):
    """
    Plots a heatmap of attention weights.
    """
    if attention_matrix is None or not tokens:
        print("Warning: No attention data or tokens provided for plotting.")
        return

    # If attention_matrix is 3D (num_heads, seq_len, seq_len), average over heads
    if len(attention_matrix.shape) == 3:
        attention_matrix = np.mean(attention_matrix, axis=0)

    # Trim tokens and matrix if they are too long for a single plot
    max_tokens = 50
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        attention_matrix = attention_matrix[:max_tokens, :max_tokens]

    plt.figure(figsize=(15, 12))
    
    # Use viridis or plasma for better visibility
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, 
                cmap='viridis', annot=False, cbar=True)
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention heatmap saved to {output_path}")
