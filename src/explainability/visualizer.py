"""
Visualization utilities for model explanations.

Provides text-based heatmaps and ranked importance reports to 
satisfy transparency requirements.
"""

NOISE_TOKENS = {
    "int", "void", "char", "return", "if", "else", 
    "for", "while", "static", "const", "{", "}", "(", ")", ";", "*", "&",
    "p", "5", "0", "1", "2", "3", "4", "6", "7", "8", "9",
}

def generate_text_heatmap(explanation: list[dict], max_tokens: int = 15) -> str:
    """Generate a clean, compact ASCII heatmap of token contributions.
    
    Args:
        explanation: List of {"token": str, "score": float} dicts.
        max_tokens: Maximum tokens to show in the heatmap.
    """
    if not explanation:
        return "No explanation available."
    
    # Filter out noise tokens and take top N
    filtered_exp = [
        item for item in explanation 
        if item["token"].lower() not in NOISE_TOKENS and len(item["token"]) > 1
    ]
    sorted_exp = sorted(filtered_exp, key=lambda x: abs(x["score"]), reverse=True)[:max_tokens]
    
    # Get max score for normalization
    max_score = max([abs(x["score"]) for x in sorted_exp]) if sorted_exp else 1.0
    if max_score == 0: max_score = 1.0
    
    lines = ["CONTRIBUTION HEATMAP (Top Token Impacts)"]
    lines.append("-" * 50)
    
    for item in sorted_exp:
        token = item["token"]
        score = item["score"]
        
        # Normalize bar length against max_score
        bar_len = int((abs(score) / max_score) * 15)
        
        if score > 0:
            indicator = "VULN"
            bar = "█" * bar_len
        elif score < 0:
            indicator = "SAFE"
            bar = "░" * bar_len
        else:
            indicator = "----"
            bar = ""
            
        lines.append(f"{token:<15} | {score:>7.2%} | {indicator:<4} {bar}")
        
    return "\n".join(lines)

def generate_outcome_summary(is_vulnerable: bool, confidence: float, explanation: list[dict]) -> tuple[str, list[str]]:
    """Generate a human-readable summary and filtered findings.
    
    Returns:
        tuple[str, list[str]]: (Formatted summary string, List of top findings)
    """
    verdict = "VULNERABLE" if is_vulnerable else "SAFE"
    summary_lines = [
        f"VERDICT: {verdict} ({confidence:.1%})",
        "\nKEY INDICATORS:"
    ]
    
    # Filter out noise tokens and sort
    filtered_indicators = [
        ind for ind in explanation 
        if ind["token"].lower() not in NOISE_TOKENS and len(ind["token"]) > 1
    ]
    
    findings = []
    # Top 3 most important informative indicators
    indicators = sorted(filtered_indicators, key=lambda x: abs(x["score"]), reverse=True)[:3]
    
    for ind in indicators:
        if ind["score"] > 0:
            finding = f"Token '{ind['token']}' strongly signals vulnerability ({ind['score']:.1%})"
        else:
            finding = f"Token '{ind['token']}' reduces vulnerability confidence ({ind['score']:.1%})"
        
        findings.append(finding)
        summary_lines.append(f"- {finding}")
            
    return "\n".join(summary_lines), findings


# ── Bias Detection ───────────────────────────────────────────────────────

# Tokens that should not be driving vulnerability predictions
SUSPICIOUS_TOKEN_PATTERNS = {
    "punctuation", "single_chars", "digits", "generic_keywords"
}

SECURITY_RELEVANT_TOKENS = {
    "free", "malloc", "strcpy", "strncpy", "sprintf", "snprintf",
    "gets", "scanf", "memcpy", "memmove", "realloc", "calloc",
    "system", "exec", "popen", "fopen", "fread", "fwrite",
    "printf", "fprintf", "buffer", "overflow", "null", "nullptr",
}


def analyze_token_bias(explanation: list[dict], top_n: int = 10) -> list[dict]:
    """Flag tokens whose high attribution scores are semantically suspicious.
    
    The reference paper warns: "the model is overly sensitive to the meanings
    of words, particularly those used as function names or variable names."
    
    Args:
        explanation: List of {"token": str, "score": float} dicts.
        top_n: Number of top tokens to analyze.
    
    Returns:
        List of warnings for suspicious tokens.
    """
    if not explanation:
        return []
    
    sorted_exp = sorted(explanation, key=lambda x: abs(x["score"]), reverse=True)[:top_n]
    
    warnings = []
    for rank, item in enumerate(sorted_exp, 1):
        token = item["token"].lower().strip()
        score = item["score"]
        
        if not token or len(token) <= 1:
            warnings.append({
                "rank": rank,
                "token": item["token"],
                "score": score,
                "concern": "Single character or empty token — likely noise, not a vulnerability indicator",
            })
        elif token in NOISE_TOKENS:
            warnings.append({
                "rank": rank,
                "token": item["token"],
                "score": score,
                "concern": "Generic C keyword or syntax — should not drive vulnerability prediction",
            })
        elif token in SECURITY_RELEVANT_TOKENS:
            # This is expected — not a warning
            continue
        elif token.isdigit():
            warnings.append({
                "rank": rank,
                "token": item["token"],
                "score": score,
                "concern": "Numeric literal — unlikely to indicate vulnerability on its own",
            })
    
    return warnings

