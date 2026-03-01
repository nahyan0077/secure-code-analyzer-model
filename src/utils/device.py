"""
Device utility for Apple Silicon MPS support.

Provides a single function to resolve the best available torch device,
preferring MPS on Apple Silicon and falling back to CPU.
"""

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU).

    Returns:
        torch.device: ``cuda`` if NVIDIA GPU available, ``mps`` if Apple 
                     Silicon GPU available, else ``cpu``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using NVIDIA CUDA backend")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS backend")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available — falling back to CPU")
    return device
