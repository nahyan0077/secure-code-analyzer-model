"""
Visualization utilities for model explanations.

Provides text-based heatmaps and ranked importance reports to 
satisfy transparency requirements.
"""

def generate_text_heatmap(explanation: list[dict], threshold: float = 0.05) -> str:
    """Generate a text-based ASCII heatmap of token contributions.
    
    Args:
        explanation: List of {"token": str, "score": float} dicts.
        threshold: Score threshold to consider a token "significant".
        
    Returns:
        str: A formatted string representing the importance.
    """
    if not explanation:
        return "No explanation available."
    
    # Sort by absolute score
    sorted_exp = sorted(explanation, key=lambda x: abs(x["score"]), reverse=True)
    
    lines = ["=== VULNERABILITY CONTRIBUTION HEATMAP ==="]
    lines.append(f"{'TOKEN':<20} | {'SCORE':<10} | {'IMPACT'}")
    lines.append("-" * 50)
    
    for item in sorted_exp:
        token = item["token"]
        score = item["score"]
        
        if score > 0:
            impact = "[!] VULNERABLE"
            # ASCII bar
            bar = "+" * int(min(score * 20, 20))
        elif score < 0:
            impact = "[ ] SAFE"
            bar = "-" * int(min(abs(score) * 20, 20))
        else:
            impact = "-"
            bar = ""
            
        lines.append(f"{token:<20} | {score:>10.4f} | {impact} {bar}")
        
    return "\n".join(lines)

def generate_outcome_summary(is_vulnerable: bool, confidence: float, explanation: list[dict]) -> str:
    """Generate a human-readable summary of the model outcome.
    
    Complies with EU AI Act requirements for understandable AI feedback.
    """
    verdict = "VULNERABLE" if is_vulnerable else "SAFE"
    summary = [
        f"MODEL VERDICT: {verdict}",
        f"CONFIDENCE: {confidence:.2%}",
        "\nKEY FINDINGS:"
    ]
    
    # Top 3 indicators
    indicators = sorted(explanation, key=lambda x: x["score"], reverse=True)[:3]
    for ind in indicators:
        if ind["score"] > 0:
            summary.append(f"- Token '{ind['token']}' increased vulnerability probability by {ind['score']:.2%}")
            
    return "\n".join(summary)
