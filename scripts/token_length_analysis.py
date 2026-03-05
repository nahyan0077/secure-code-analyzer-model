"""
Token Length Analysis for the DiverseVul dataset.

Analyzes the distribution of tokenized function lengths to understand
the impact of the MAX_LENGTH truncation setting.

Usage::

    python scripts/token_length_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import Config
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import clean_data
from src.model.model_loader import load_tokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_token_lengths(max_samples: int | None = None) -> dict:
    """Analyze token lengths across the dataset.

    Returns:
        dict: Statistics including percentiles and truncation percentages.
    """
    Config.ensure_dirs()

    # Load data
    logger.info("Loading dataset for token length analysis ...")
    df = load_dataset(Config.DATA_PATH, max_samples=max_samples or Config.MAX_SAMPLES)
    df = clean_data(df)

    # Tokenize
    tokenizer = load_tokenizer()
    logger.info("Tokenizing %d functions (this may take a while) ...", len(df))

    lengths = []
    for i, text in enumerate(df["func"].tolist()):
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)
        lengths.append(len(tokens["input_ids"]))
        if (i + 1) % 10000 == 0:
            logger.info("  Tokenized %d / %d ...", i + 1, len(df))

    lengths = np.array(lengths)

    # Compute stats
    stats = {
        "total_functions": len(lengths),
        "mean_tokens": round(float(np.mean(lengths)), 1),
        "median_tokens": int(np.median(lengths)),
        "std_tokens": round(float(np.std(lengths)), 1),
        "min_tokens": int(np.min(lengths)),
        "max_tokens": int(np.max(lengths)),
        "percentiles": {
            "p50": int(np.percentile(lengths, 50)),
            "p75": int(np.percentile(lengths, 75)),
            "p90": int(np.percentile(lengths, 90)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
        },
        "truncation_rates": {
            "at_128": round(float((lengths > 128).mean() * 100), 2),
            "at_256": round(float((lengths > 256).mean() * 100), 2),
            "at_512": round(float((lengths > 512).mean() * 100), 2),
        },
        "within_limits": {
            "under_128": round(float((lengths <= 128).mean() * 100), 2),
            "under_256": round(float((lengths <= 256).mean() * 100), 2),
            "under_512": round(float((lengths <= 512).mean() * 100), 2),
        },
    }

    # Save
    output_path = Config.REPORTS_DIR / "token_length_analysis.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("TOKEN LENGTH ANALYSIS")
    logger.info("=" * 60)
    logger.info("  Total functions:  %d", stats["total_functions"])
    logger.info("  Mean tokens:      %.1f", stats["mean_tokens"])
    logger.info("  Median tokens:    %d", stats["median_tokens"])
    logger.info("  P95 tokens:       %d", stats["percentiles"]["p95"])
    logger.info("  P99 tokens:       %d", stats["percentiles"]["p99"])
    logger.info("  Max tokens:       %d", stats["max_tokens"])
    logger.info("")
    logger.info("  Truncated at 256: %.1f%% of functions", stats["truncation_rates"]["at_256"])
    logger.info("  Truncated at 512: %.1f%% of functions", stats["truncation_rates"]["at_512"])
    logger.info("=" * 60)
    logger.info("Results saved to %s", output_path)

    return stats


if __name__ == "__main__":
    analyze_token_lengths()
