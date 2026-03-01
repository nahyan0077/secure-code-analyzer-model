"""
Data preprocessing for the vulnerability detection pipeline.

Provides cleaning, balancing, splitting, class-weight computation,
and a PyTorch Dataset that tokenises code snippets with CodeBERT.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from configs.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Cleaning ─────────────────────────────────────────────────────────────


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null ``func`` or ``target`` and remove duplicates.

    Args:
        df: Raw dataset DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    before = len(df)
    logger.info("Starting data cleaning [dim](dropping nulls & duplicates)[/dim] ...")
    
    df = df.dropna(subset=["func", "target"])
    df = df.drop_duplicates(subset=["func"])
    df = df.reset_index(drop=True)
    
    dropped = before - len(df)
    logger.info(
        "Cleaning finished: [bold green]%d[/bold green] rows remaining [dim](dropped %d)[/dim]", 
        len(df), dropped
    )
    return df


# ── Balancing ────────────────────────────────────────────────────────────


def balance_data(df: pd.DataFrame, strategy: str = "undersample") -> pd.DataFrame:
    """Balance the dataset by under-sampling the majority class.

    This is an **optional** step — the default training pipeline uses
    class-weighted loss instead, which is generally preferred.

    Args:
        df: Cleaned DataFrame.
        strategy: ``"undersample"    """
    logger.info("Applying minority-class balancing [dim](strategy=%s)[/dim] ...", strategy)
    vuln = df[df["target"] == 1]
    safe = df[df["target"] == 0]
    minority_size = min(len(vuln), len(safe))

    if strategy == "undersample":
        majority = safe if len(safe) > len(vuln) else vuln
        majority_down = majority.sample(n=minority_size, random_state=Config.TRAIN_PARAMS["seed"])
        minority = vuln if len(vuln) <= len(safe) else safe
        balanced = pd.concat([majority_down, minority]).sample(
            frac=1, random_state=Config.TRAIN_PARAMS["seed"]
        )
        balanced = balanced.reset_index(drop=True)
        logger.info("Balanced data: [bold green]%d[/bold green] → [bold yellow]%d[/bold yellow] rows", len(df), len(balanced))
        return balanced

    raise ValueError(f"Unknown balancing strategy: {strategy}")


# ── Splitting ────────────────────────────────────────────────────────────


def split_data(
    df: pd.DataFrame,
    test_size: float = Config.TEST_SIZE,
    val_size: float = Config.VAL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split.

    Args:
        df: Cleaned (or balanced) DataFrame.
        test_size: Fraction for the test set.
        val_size: Fraction for the validation set (relative to total).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info("Splitting data into [bold cyan]Train/Val/Test[/bold cyan] sets ...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target"],
        random_state=Config.TRAIN_PARAMS["seed"],
    )
    # val_size is relative to the remaining train set
    relative_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val,
        stratify=train_df["target"],
        random_state=Config.TRAIN_PARAMS["seed"],
    )
    logger.info(
        "Split complete: train=[bold green]%d[/bold green], val=[bold green]%d[/bold green], test=[bold green]%d[/bold green]",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


# ── Class Weights ────────────────────────────────────────────────────────


def compute_class_weights(
    df: pd.DataFrame,
    num_classes: int = Config.NUM_LABELS,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted loss.

    Always returns a tensor of shape ``(num_classes,)`` even if some
    classes are missing from the data.

    Args:
        df: DataFrame with ``target`` column.
        num_classes: Total number of classes.

    Returns:
        torch.Tensor: Weights tensor of shape ``(num_classes,)``.
    """
    logger.info("Computing [bold yellow]class weights[/bold yellow] for addressing imbalance ...")
    counts = df["target"].value_counts()
    total = len(df)
    weights = []
    for cls in range(num_classes):
        if cls in counts and counts[cls] > 0:
            weights.append(total / (num_classes * counts[cls]))
        else:
            logger.warning("Class [bold red]%d[/bold red] missing from dataset; using default weight 1.0", cls)
            weights.append(1.0)  # neutral weight for missing classes
    weights = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights resolved: %s", weights.tolist())
    return weights


# ── PyTorch Dataset ──────────────────────────────────────────────────────


class VulnerabilityDataset(Dataset):
    """PyTorch Dataset for code vulnerability classification.

    Tokenises ``func`` text with the CodeBERT tokenizer and returns
    tensors ready for ``AutoModelForSequenceClassification``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = Config.MAX_LENGTH,
    ) -> None:
        """
        Args:
            df: DataFrame with ``func`` and ``target`` columns.
            tokenizer: HuggingFace tokenizer. Defaults to CodeBERT tokenizer.
            max_length: Maximum token length.
        """
        self.texts = df["func"].tolist()
        self.labels = df["target"].tolist()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
