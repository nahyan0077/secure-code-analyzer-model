import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.explainability.shap_explainer import ShapExplainer
from configs.config import Config

def test_shap_plot():
    # Use a dummy code snippet for testing
    code_snippet = """
    #include <stdio.h>
    int main() {
        char buffer[10];
        gets(buffer); // Vulnerable
        return 0;
    }
    """
    
    # Initialize explainer
    # Note: This might take a while to load the model
    try:
        explainer = ShapExplainer()
        
        output_plot = "reports/plots/test_shap_importance.png"
        print(f"Generating SHAP plot for test snippet...")
        
        explainer.explain(code_snippet, max_tokens=15, output_plot_path=output_plot)
        
        if os.path.exists(output_plot):
            print(f"Success: SHAP plot generated at {output_plot}")
        else:
            print(f"Failure: SHAP plot not found at {output_plot}")
            
    except Exception as e:
        print(f"Error during SHAP test: {e}")

if __name__ == "__main__":
    if not os.path.exists("reports/plots"):
        os.makedirs("reports/plots")
    test_shap_plot()
