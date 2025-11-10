"""Script to download the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.paths import RAW_DATA_DIR


def main():
    """Download dataset."""
    print("Dataset download script")
    print(f"Target directory: {RAW_DATA_DIR}")
    print("\n⚠️  Please implement dataset download logic here.")
    print("You can use:")
    print("  - kaggle API")
    print("  - wget/curl")
    print("  - Custom download script")


if __name__ == "__main__":
    main()
