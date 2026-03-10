import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.utils.visualizer import plot_training_history, plot_confusion_matrix

def main():
    reports_dir = "reports"
    plots_dir = os.path.join(reports_dir, "plots")
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    # Generate training evolution plot
    history_path = os.path.join(reports_dir, "training_history.json")
    if os.path.exists(history_path):
        plot_training_history(history_path, plots_dir)
    else:
        print(f"Skipping training evolution: {history_path} not found.")

    # Generate confusion matrix heatmap
    metrics_path = os.path.join(reports_dir, "metrics.json")
    if os.path.exists(metrics_path):
        plot_confusion_matrix(metrics_path, plots_dir)
    else:
        print(f"Skipping confusion matrix: {metrics_path} not found.")

    print("\nAll plots generated successfully in: " + plots_dir)

if __name__ == "__main__":
    main()
