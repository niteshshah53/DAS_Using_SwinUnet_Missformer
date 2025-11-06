import re
import os
from pathlib import Path

def extract_metrics_from_out_file(file_path):
    """
    Extract mean metrics for each manuscript from a .out file.
    
    Args:
        file_path: Path to the .out file
        
    Returns:
        Dictionary mapping manuscript names to their metrics
    """
    manuscripts = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match manuscript name and metrics
    # Look for "Manuscript: ManuscriptName" followed by "Mean metrics:" section
    # Handle both "TESTING:" and "Manuscript:" prefixes
    pattern = r'(?:TESTING:|Manuscript:)\s+(\w+).*?Mean metrics:.*?Mean Precision:\s+([\d.]+).*?Mean Recall:\s+([\d.]+).*?Mean F1:\s+([\d.]+).*?Mean IoU:\s+([\d.]+)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        manuscript_name = match.group(1).lower()
        precision = float(match.group(2))
        recall = float(match.group(3))
        f1 = float(match.group(4))
        iou = float(match.group(5))
        
        manuscripts[manuscript_name] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou
        }
    
    return manuscripts


def calculate_averages(manuscripts):
    """
    Calculate average metrics across all manuscripts.
    
    Args:
        manuscripts: Dictionary mapping manuscript names to metrics
        
    Returns:
        Dictionary with average metrics
    """
    if not manuscripts:
        return {}
    
    metrics = ["Precision", "Recall", "F1", "IoU"]
    averages = {metric: 0.0 for metric in metrics}
    
    # Sum all metric values
    for manuscript in manuscripts.values():
        for metric in metrics:
            averages[metric] += manuscript[metric]
    
    # Divide by number of manuscripts to get averages
    num_manuscripts = len(manuscripts)
    for metric in metrics:
        averages[metric] /= num_manuscripts
    
    return averages


def main():
    # Path to the .out file in a1 folder
    # Adjust the path as needed based on your directory structure
    script_dir = Path(__file__).parent
    out_file = script_dir / "Results" / "a2" / "baseline_smart_skip_ds_groupnorm_1307088.out"
    
    # If running from models/network directory, use:
    # out_file = Path("a1") / "bl_ash_ds_fff_bo_fl_1294128.out"
    
    if not out_file.exists():
        print(f"Error: File not found: {out_file}")
        print("Please check the file path.")
        return
    
    print(f"Reading metrics from: {out_file}\n")
    
    # Extract metrics from the file
    manuscripts = extract_metrics_from_out_file(out_file)
    
    if not manuscripts:
        print("No metrics found in the file.")
        return
    
    # Print extracted metrics
    print("Extracted Metrics from .out file:")
    print("=" * 60)
    for name, metrics in manuscripts.items():
        print(f"\n{name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Calculate averages
    averages = calculate_averages(manuscripts)
    
    # Print the results
    print("\n" + "=" * 60)
    print("Average Metrics Across All Manuscripts:")
    print("=" * 60)
    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nTotal manuscripts processed: {len(manuscripts)}")


if __name__ == "__main__":
    main()