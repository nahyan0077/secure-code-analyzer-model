"""
Central configuration for the Vulnerability Detection system.

All paths, model parameters, and training hyperparameters are defined here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the vulnerability detection pipeline."""

    # ── Paths ────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = PROJECT_ROOT.parent / "data" / "diversevul_20230702.json"
    MODEL_DIR: Path = PROJECT_ROOT / "model"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    # ── Model ────────────────────────────────────────────────────────────
    MODEL_NAME: str = "microsoft/codebert-base"
    NUM_LABELS: int = 2
    MAX_LENGTH: int = 512

    # ── Training ─────────────────────────────────────────────────────────
    TRAIN_PARAMS: dict = {
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 50,
        "seed": 42,
    }

    # ── Data ─────────────────────────────────────────────────────────────
    MAX_SAMPLES: int | None = int(v) if (v := os.getenv("MAX_SAMPLES")) and v.strip() not in ("", "0") else None
    TEST_SIZE: float = 0.1
    VAL_SIZE: float = 0.1

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create output directories if they don't exist."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
