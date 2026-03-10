"""
Model loading and persistence utilities.

Wraps HuggingFace ``AutoModelForSequenceClassification`` and
``AutoTokenizer`` for the optimized vulnerability classifier.
"""

from pathlib import Path
from typing import Tuple

import json
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from configs.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(
    model_name: str = Config.MODEL_NAME,
    num_labels: int = Config.NUM_LABELS,
) -> PreTrainedModel:
    """Load a pretrained model with a sequence-classification head.

    Args:
        model_name: HuggingFace model identifier or local path.
        num_labels: Number of output labels.

    Returns:
        PreTrainedModel: Model ready for training or inference.
    """
    logger.info("Loading model: %s (num_labels=%d)", model_name, num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={0: "safe", 1: "vulnerable"},
        label2id={"safe": 0, "vulnerable": 1},
    )

    # Inject a stronger MLP head for natural language models
    if "codebert" not in model_name.lower():
        logger.info("Injecting [bold yellow]stronger MLP classifier head[/bold yellow] for Optimized BERT model")
        hidden_size = model.config.hidden_size
        model.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
    return model


def load_tokenizer(
    model_name: str = Config.MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer for the given model.

    Args:
        model_name: HuggingFace model identifier or local path.

    Returns:
        PreTrainedTokenizerBase: Tokenizer instance.
    """
    logger.info("Loading tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    path: Path | str = Config.MODEL_DIR,
) -> None:
    """Save model and tokenizer to disk.

    Args:
        model: Trained model.
        tokenizer: Corresponding tokenizer.
        path: Directory to save into.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info("Model and tokenizer saved to %s", path)


def load_trained_model(
    path: Path | str = Config.MODEL_DIR,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a previously saved model and tokenizer.

    Args:
        path: Directory containing saved model files.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        FileNotFoundError: If the model directory doesn't exist.
    """
    path = Path(path)
    config_file = path / "config.json"
    if not path.exists() or not config_file.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. "
            f"Please run training first: python -m src.model.train"
        )
    logger.info("Loading trained model from %s", path)
    
    # Check what the base model name was from the saved config
    with open(config_file, "r") as f:
        saved_config = json.load(f)
    base_model_name_or_path = saved_config.get("_name_or_path", Config.MODEL_NAME)

    # 1. Load the model architecture
    model = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="eager")
    
    # 2. Re-inject the custom MLP head if it's a non-CodeBERT model
    # We use base_model_name_or_path to determine if it needs the head
    if "codebert" not in base_model_name_or_path.lower():
        logger.info("Re-injecting [bold yellow]MLP head[/bold yellow] for loading trained weights")
        hidden_size = model.config.hidden_size
        num_labels = len(model.config.id2label)
        model.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        # Now reload the weights specifically to map into the new classifier
        from safetensors.torch import load_model as load_safe_weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            load_safe_weights(model, weights_path, strict=False)
        else:
            # Fallback for bin files if safetensors doesn't exist
            bin_path = path / "pytorch_model.bin"
            if bin_path.exists():
                model.load_state_dict(torch.load(bin_path, map_location="cpu"), strict=False)

    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
