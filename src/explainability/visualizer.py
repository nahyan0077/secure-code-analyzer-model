"""
Visualization utilities for model explanations.

Provides text-based heatmaps and ranked importance reports to 
satisfy transparency requirements.
"""

def generate_text_heatmap(explanation: list[dict], max_tokens: int = 10) -> str:
    """Generate a clean, compact ASCII heatmap of token contributions.
    
    Args:
        explanation: List of {"token": str, "score": float} dicts.
        max_tokens: Maximum tokens to show in the heatmap.
        
    Returns:
        str: A formatted string representing the importance.
    """
    if not explanation:
        return "No explanation available."
    
    # Sort by absolute score and take top N
    sorted_exp = sorted(explanation, key=lambda x: abs(x["score"]), reverse=True)[:max_tokens]
    
    lines = ["CONTRIBUTION HEATMAP (Top Token Impacts)"]
    lines.append("-" * 50)
    
    for item in sorted_exp:
        token = item["token"]
        score = item["score"]
        
        # Use simple indicators and a cleaner bar
        if score > 0:
            indicator = "VULN"
            # Use block characters for a premium feel if possible, or simple '='
            bar_len = int(min(score * 30, 15))
            bar = "█" * bar_len
        elif score < 0:
            indicator = "SAFE"
            bar_len = int(min(abs(score) * 30, 15))
            bar = "░" * bar_len
        else:
            indicator = "----"
            bar = ""
            
        lines.append(f"{token:<15} | {score:>7.2%} | {indicator:<4} {bar}")
        
    return "\n".join(lines)

def generate_outcome_summary(is_vulnerable: bool, confidence: float, explanation: list[dict]) -> tuple[str, list[str]]:
    """Generate a human-readable summary and a list of structured findings.
    
    Returns:
        tuple[str, list[str]]: (Formatted summary string, List of top findings)
    """
    verdict = "VULNERABLE" if is_vulnerable else "SAFE"
    summary_lines = [
        f"VERDICT: {verdict} ({confidence:.1%})",
        "\nKEY INDICATORS:"
    ]
    
    findings = []
    # Top 3 most important indicators
    indicators = sorted(explanation, key=lambda x: abs(x["score"]), reverse=True)[:3]
    
    for ind in indicators:
        impact = "vulnerability" if ind["score"] > 0 else "safety"
        finding = f"Token '{ind['token']}' strongly signals {impact} ({ind['score']:.1%})"
        findings.append(finding)
        summary_lines.append(f"- {finding}")
            
    return "\n".join(summary_lines), findings
