"""⭐ EDA Analysis Script - First Step."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from eda.dataset_analysis import analyze_dataset_structure
from eda.class_distribution import analyze_class_distribution, calculate_class_weights
import json


def main():
    """Run comprehensive EDA analysis."""
    print("=" * 60)
    print("FLOORPLAN DATASET - EDA ANALYSIS")
    print("=" * 60)
    
    data_dir = Path("data/raw")
    
    print("\n1. Dataset Structure Analysis")
    print("-" * 60)
    structure_info = analyze_dataset_structure(data_dir)
    print(json.dumps(structure_info, indent=2))
    
    print("\n2. Class Distribution Analysis")
    print("-" * 60)
    mask_dir = data_dir / "masks"
    if mask_dir.exists():
        class_info = analyze_class_distribution(mask_dir)
        print(json.dumps(class_info, indent=2))
        
        weights = calculate_class_weights(class_info['class_counts'])
        print("\n3. Calculated Class Weights")
        print("-" * 60)
        print(json.dumps(weights, indent=2))
    else:
        print("Mask directory not found. Please download dataset first.")
    
    print("\n" + "=" * 60)
    print("EDA Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
