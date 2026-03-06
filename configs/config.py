"""
Central configuration for the Vulnerability Detection system.

All paths, model parameters, and training hyperparameters are defined here.
Auto-detects GPU vs MPS and adjusts training params accordingly.
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


def _detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Config:
    """Central configuration for the vulnerability detection pipeline."""

    # ── Device Detection ─────────────────────────────────────────────────
    DEVICE: str = _detect_device()
    IS_GPU: bool = DEVICE == "cuda"

    # ── Paths ────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = Path(os.getenv("DATA_PATH", str(PROJECT_ROOT.parent / "data" / "diversevul_20230702.json")))
    MODEL_DIR: Path = PROJECT_ROOT / "model"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    # ── Model ────────────────────────────────────────────────────────────
    MODEL_NAME: str = "bert-base-uncased"
    NUM_LABELS: int = 2
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "512" if IS_GPU else "256"))

    # ── Training (auto-adjusts for GPU vs MPS) ───────────────────────────
    TRAIN_PARAMS: dict = {
        "epochs": 10,
        "batch_size": int(os.getenv("BATCH_SIZE", "16" if IS_GPU else "1")),
        "gradient_accumulation_steps": 1 if IS_GPU else 8,  # GPU: large batch directly; MPS: accumulate
        "learning_rate": 3e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 50,
        "seed": 42,
        "early_stopping_patience": 2,
    }

    # ── Data ─────────────────────────────────────────────────────────────
    MAX_SAMPLES: int | None = int(v) if (v := os.getenv("MAX_SAMPLES")) and v.strip() not in ("", "0") else None
    MAX_VAL_SAMPLES: int | None = None if IS_GPU else 2000  # GPU has enough memory for full val set
    TEST_SIZE: float = 0.1
    VAL_SIZE: float = 0.1

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create output directories if they don't exist."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
