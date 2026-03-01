"""
Dataset loader for DiverseVul JSONL data.

Reads the JSON Lines file, validates required columns, and returns
a pandas DataFrame ready for preprocessing.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_COLUMNS = {"func", "target"}


def load_dataset(
    path: Path | str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Load the DiverseVul JSONL dataset.

    Args:
        path: Path to the ``.json`` / ``.jsonl`` file.
        max_samples: If set and > 0, load at most this many records.
            ``None`` or ``0`` loads all records.

    Returns:
        pd.DataFrame: DataFrame with at least ``func`` and ``target`` columns.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Reading raw JSONL records from [bold cyan]%s[/bold cyan] ...", path)

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            
            if (i + 1) % 50000 == 0:
                logger.info("Parsed %d records ...", i + 1)

    logger.info("Finished parsing [bold green]%d[/bold green] raw records.", len(records))
    df = pd.DataFrame(records)

    # Validate columns
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error("Validation failed: missing columns %s", missing)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Shuffle to avoid ordering bias (JSONL may be sorted by label)
    logger.info("Shuffling dataset to eliminate ordering bias ...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Subsample AFTER shuffling so we get a representative class mix
    if max_samples and max_samples > 0 and len(df) > max_samples:
        logger.info("Subsampling to [bold yellow]%d[/bold yellow] records ...", max_samples)
        df = df.head(max_samples).reset_index(drop=True)

    # Keep only what we need + useful metadata
    keep_cols = [c for c in ["func", "target", "cwe", "project"] if c in df.columns]
    df = df[keep_cols]

    vuln_count = (df["target"] == 1).sum()
    safe_count = (df["target"] == 0).sum()
    logger.info(
        "Loading complete: [bold green]%d[/bold green] records [dim](%d vuln, %d safe)[/dim]",
        len(df), vuln_count, safe_count
    )
    return df
