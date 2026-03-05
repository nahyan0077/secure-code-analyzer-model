"""
Global SHAP Explainability Analysis.

Runs SHAP on a sample of the dataset to produce global token importance
rankings and flag potentially suspicious/biased tokens.

Usage::

    python -m src.explainability.global_explainer
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import Config
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import clean_data, split_data
from src.explainability.shap_explainer import ShapExplainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tokens that should NOT have high attribution scores
SUSPICIOUS_CATEGORIES = {
    "punctuation": set("( ) { } [ ] ; , . : < > = + - * / & | ^ ~ ! ? # %".split()),
    "single_chars": set("a b c d e f g h i j k l m n o p q r s t u v w x y z".split()),
    "digits": set("0 1 2 3 4 5 6 7 8 9".split()),
    "c_keywords": {
        "int", "void", "char", "return", "if", "else", "for", "while",
        "do", "switch", "case", "break", "continue", "static", "const",
        "unsigned", "long", "short", "float", "double", "struct", "typedef",
        "enum", "extern", "register", "volatile", "inline",
    },
}


def run_global_analysis(sample_size: int = 50) -> dict:
    """Run SHAP on a sample and produce global importance rankings.

    Args:
        sample_size: Number of test examples to analyze.

    Returns:
        dict: Global analysis results with rankings and bias flags.
    """
    Config.ensure_dirs()

    # Load test data
    logger.info("Loading dataset for global explainability analysis ...")
    df = load_dataset(Config.DATA_PATH, max_samples=Config.MAX_SAMPLES)
    df = clean_data(df)
    _, _, test_df = split_data(df)

    # Sample from test set
    sample_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    logger.info("Analyzing %d test samples with SHAP ...", len(sample_df))

    # Initialize explainer
    explainer = ShapExplainer()

    # Aggregate token scores
    token_scores: Counter = Counter()
    token_counts: Counter = Counter()

    for i, (_, row) in enumerate(sample_df.iterrows()):
        try:
            scores = explainer.explain(row["func"], max_tokens=100)
            for item in scores:
                token = item["token"].lower().strip()
                if token:
                    token_scores[token] += abs(item["score"])
                    token_counts[token] += 1
        except Exception as e:
            logger.warning("SHAP failed for sample %d: %s", i, e)

        if (i + 1) % 10 == 0:
            logger.info("  Processed %d / %d samples ...", i + 1, len(sample_df))

    # Compute average absolute SHAP values (global importance)
    global_importance = {}
    for token in token_scores:
        global_importance[token] = {
            "avg_abs_shap": round(token_scores[token] / token_counts[token], 6),
            "total_abs_shap": round(token_scores[token], 6),
            "occurrences": token_counts[token],
        }

    # Sort by average absolute SHAP
    ranked = sorted(global_importance.items(), key=lambda x: x[1]["avg_abs_shap"], reverse=True)

    # Flag suspicious tokens in top 30
    top_30 = ranked[:30]
    suspicious_flags = []
    all_suspicious = set()
    for cat_name, cat_tokens in SUSPICIOUS_CATEGORIES.items():
        all_suspicious.update(cat_tokens)

    for rank, (token, stats) in enumerate(top_30, 1):
        for cat_name, cat_tokens in SUSPICIOUS_CATEGORIES.items():
            if token in cat_tokens:
                suspicious_flags.append({
                    "rank": rank,
                    "token": token,
                    "category": cat_name,
                    "avg_abs_shap": stats["avg_abs_shap"],
                    "concern": f"Token '{token}' ({cat_name}) should not have high attribution — possible model bias",
                })

    # Build results
    results = {
        "analysis_config": {
            "sample_size": len(sample_df),
            "unique_tokens_analyzed": len(global_importance),
        },
        "top_30_global_tokens": [
            {"rank": i + 1, "token": t, **s} for i, (t, s) in enumerate(top_30)
        ],
        "suspicious_tokens": suspicious_flags,
        "bias_summary": {
            "suspicious_in_top_30": len(suspicious_flags),
            "verdict": "POTENTIAL BIAS DETECTED" if len(suspicious_flags) > 5 else "ACCEPTABLE",
        },
    }

    # Save
    output_path = Config.REPORTS_DIR / "global_explanations.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Log summary
    logger.info("=" * 60)
    logger.info("GLOBAL SHAP ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info("  Samples analyzed:    %d", len(sample_df))
    logger.info("  Unique tokens:       %d", len(global_importance))
    logger.info("")
    logger.info("  Top 10 Global Tokens:")
    for i, (token, stats) in enumerate(ranked[:10], 1):
        flag = " ⚠️ SUSPICIOUS" if token in all_suspicious else ""
        logger.info("    %2d. %-15s  avg|SHAP|=%.4f  (seen %d times)%s",
                     i, token, stats["avg_abs_shap"], stats["occurrences"], flag)
    logger.info("")
    logger.info("  Suspicious tokens in top 30: %d", len(suspicious_flags))
    if suspicious_flags:
        for sf in suspicious_flags:
            logger.info("    ⚠️  Rank %d: '%s' (%s)", sf["rank"], sf["token"], sf["category"])
    logger.info("=" * 60)
    logger.info("Full results saved to %s", output_path)

    return results


if __name__ == "__main__":
    run_global_analysis()
