"""
Proper CubiCasa5K Dataset Conversion Script
Converts CubiCasa5K structure to standard format for EDA and training
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_cubicasa_to_standard(source_dir, target_dir):
    """
    Convert CubiCasa5K dataset to standard format
    
    CubiCasa5K Structure:
        cubicasa5k/high_quality/
        ├── 103/
        │   ├── F1_original.png    # Floor plan image
        │   ├── F1_scaled.png      # Scaled version
        │   └── model.svg          # Annotation (SVG format)
        ├── 107/
        └── ...
    
    Target Structure:
        target_dir/
        ├── images/
        │   ├── 103.png
        │   ├── 107.png
        │   └── ...
        └── annotations/
            ├── 103.png
            ├── 107.png
            └── ...
    
    Note: CubiCasa5K provides SVG annotations, but for semantic segmentation
    we need rasterized PNG masks. This script will look for pre-rendered PNG
    versions or you'll need to render them separately.
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create output directories
    images_dir = target_path / "images"
    annotations_dir = target_path / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print("="*80)
    
    # Get all subdirectories (each represents one floor plan)
    floor_plan_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(floor_plan_dirs)} floor plan directories")
    
    copied_images = 0
    copied_annotations = 0
    missing_annotations = []
    
    # Process each floor plan directory
    for floor_dir in tqdm(floor_plan_dirs, desc="Converting"):
        floor_id = floor_dir.name
        
        # Copy image (F1_original.png)
        image_source = floor_dir / "F1_original.png"
        if image_source.exists():
            image_target = images_dir / f"{floor_id}.png"
            shutil.copy2(image_source, image_target)
            copied_images += 1
        else:
            print(f"⚠️ Missing image: {floor_id}")
        
        # Look for PNG annotation
        # CubiCasa5K might have rendered PNG versions in some cases
        annotation_candidates = [
            floor_dir / "model.png",
            floor_dir / "model_segmentation.png",
            floor_dir / "F1_segmentation.png",
        ]
        
        annotation_found = False
        for ann_candidate in annotation_candidates:
            if ann_candidate.exists():
                annotation_target = annotations_dir / f"{floor_id}.png"
                shutil.copy2(ann_candidate, annotation_target)
                copied_annotations += 1
                annotation_found = True
                break
        
        if not annotation_found:
            missing_annotations.append(floor_id)
    
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"✓ Copied {copied_images} images to {images_dir}")
    print(f"✓ Copied {copied_annotations} annotations to {annotations_dir}")
    
    if missing_annotations:
        print(f"\n⚠️ WARNING: {len(missing_annotations)} annotations NOT found (SVG only)")
        print(f"  These floor plans have model.svg but no PNG version")
        print(f"  First 10: {missing_annotations[:10]}")
        print(f"\n  SOLUTION: You need to render SVG to PNG masks")
        print(f"  See documentation for SVG conversion instructions")
    
    print(f"\nDataset ready at: {target_path}")
    print(f"Run EDA with: python src\\eda\\eda_analysis.py --dataset_path {target_path}")
    

def check_cubicasa_structure(source_dir):
    """
    Check and report CubiCasa5K structure
    """
    source_path = Path(source_dir)
    
    print("CHECKING CUBICASA5K STRUCTURE")
    print("="*80)
    
    if not source_path.exists():
        print(f"❌ Directory not found: {source_path}")
        return False
    
    # Check for floor plan directories
    floor_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not floor_dirs:
        print(f"❌ No floor plan directories found in {source_path}")
        return False
    
    print(f"✓ Found {len(floor_dirs)} floor plan directories")
    
    # Sample first directory
    sample_dir = floor_dirs[0]
    print(f"\nSample directory: {sample_dir.name}")
    print("Contents:")
    
    files = list(sample_dir.iterdir())
    for f in files:
        size_mb = f.stat().st_size / (1024*1024) if f.is_file() else 0
        print(f"  - {f.name:<30} {size_mb:>8.2f} MB")
    
    # Check for annotation formats
    has_svg = (sample_dir / "model.svg").exists()
    has_png = any((sample_dir / name).exists() for name in ["model.png", "model_segmentation.png"])
    
    print(f"\nAnnotation Format:")
    print(f"  SVG annotations: {'✓ YES' if has_svg else '✗ NO'}")
    print(f"  PNG annotations: {'✓ YES' if has_png else '✗ NO'}")
    
    if has_svg and not has_png:
        print(f"\n⚠️ WARNING: Only SVG annotations found!")
        print(f"  You need to convert SVG to PNG for semantic segmentation")
        print(f"  Options:")
        print(f"    1. Use CubiCasa5K's provided tools to render SVGs")
        print(f"    2. Use external SVG-to-PNG conversion tools")
        print(f"    3. Download pre-rendered version (if available)")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CubiCasa5K to standard format")
    parser.add_argument(
        "--source",
        type=str,
        default="data/cubicasa5k/cubicasa5k/high_quality",
        help="Path to CubiCasa5K high_quality directory"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/cubicasa5k_converted",
        help="Path to output directory"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check structure, don't convert"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        check_cubicasa_structure(args.source)
    else:
        # First check structure
        if check_cubicasa_structure(args.source):
            print("\n" + "="*80)
            input("Press Enter to continue with conversion...")
            convert_cubicasa_to_standard(args.source, args.target)
