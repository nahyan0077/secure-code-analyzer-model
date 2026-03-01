"""
Model loading and persistence utilities.

Wraps HuggingFace ``AutoModelForSequenceClassification`` and
``AutoTokenizer`` for the CodeBERT-based vulnerability classifier.
"""

from pathlib import Path
from typing import Tuple

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
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
