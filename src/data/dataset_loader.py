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


def load_dataset_indices(path: Path | str) -> pd.DataFrame:
    """Pass 1: Load ONLY labels and metadata (no code) to minimize RAM.
    
    Args:
        path: Path to JSONL file.
        
    Returns:
        pd.DataFrame: DataFrame with metadata but NO 'func' column.
    """
    path = Path(path)
    logger.info("Pass 1: Scanning metadata from [bold cyan]%s[/bold cyan] ...", path)
    
    indices_data = []
    # Only keep metadata for splitting/balancing
    keep_keys = {"target", "cwe", "project"}
    
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            
            # Fast parse: only look for metadata
            raw = json.loads(line)
            record = {k: raw[k] for k in keep_keys if k in raw}
            record["file_index"] = i # Store original line index
            indices_data.append(record)
            
            if (i + 1) % 100000 == 0:
                logger.info("Scanned metadata for %d records ...", i + 1)
                
    df = pd.DataFrame(indices_data)
    logger.info("Pass 1 complete: Metadata for [bold green]%d[/bold green] records loaded.", len(df))
    return df


def load_selected_records(path: Path | str, selected_indices: set[int]) -> pd.DataFrame:
    """Pass 2: Selective load. Only parse full JSON for needed indices.
    
    Args:
        path: Path to JSONL file.
        selected_indices: Set of line indices to load.
        
    Returns:
        pd.DataFrame: Final DataFrame with 'func' and 'target'.
    """
    path = Path(path)
    logger.info("Pass 2: Selective loading [bold yellow]%d[/bold yellow] snippets ...", len(selected_indices))
    
    records = []
    keep_keys = {"func", "target", "cwe", "project"}
    
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i not in selected_indices:
                continue
                
            line = line.strip()
            if not line:
                continue
                
            raw = json.loads(line)
            filtered = {k: raw[k] for k in keep_keys if k in raw}
            records.append(filtered)
            
            if len(records) % 10000 == 0:
                logger.info("Loaded %d/%d code snippets ...", len(records), len(selected_indices))
                
    df = pd.DataFrame(records)
    logger.info("Pass 2 complete: [bold green]%d[/bold green] records loaded into memory.", len(df))
    return df


def load_dataset(path: Path | str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """Legacy wrapper that uses two-pass internal logic."""
    df_indices = load_dataset_indices(path)
    
    if max_samples and max_samples > 0 and len(df_indices) > max_samples:
        df_indices = df_indices.sample(n=max_samples, random_state=42)
        
    indices_to_load = set(df_indices["file_index"].tolist())
    return load_selected_records(path, indices_to_load)

