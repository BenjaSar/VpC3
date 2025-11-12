"""
SVG to PNG Converter for CubiCasa5K Annotations
Converts model.svg files to semantic segmentation PNG masks
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False
    print("⚠️ cairosvg not installed. Install with: pip install cairosvg")

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    HAS_SVGLIB = True
except ImportError:
    HAS_SVGLIB = False
    print("⚠️ svglib not installed. Install with: pip install svglib reportlab")


def convert_svg_to_png_cairosvg(svg_path, png_path, width=None, height=None):
    """
    Convert SVG to PNG using cairosvg (preferred method)
    """
    try:
        if width and height:
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(png_path),
                output_width=width,
                output_height=height
            )
        else:
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        return True
    except Exception as e:
        print(f"Error with cairosvg: {e}")
        return False


def convert_svg_to_png_svglib(svg_path, png_path, width=None, height=None):
    """
    Convert SVG to PNG using svglib (alternative method)
    """
    try:
        drawing = svg2rlg(str(svg_path))
        if drawing:
            if width and height:
                drawing.width = width
                drawing.height = height
            renderPM.drawToFile(drawing, str(png_path), fmt="PNG")
            return True
        return False
    except Exception as e:
        print(f"Error with svglib: {e}")
        return False


def convert_cubicasa_svg_annotations(source_dir, target_dir, max_samples=None):
    """
    Convert all CubiCasa5K SVG annotations to PNG
    
    Args:
        source_dir: Path to cubicasa5k/high_quality directory
        target_dir: Path to output directory
        max_samples: Optional limit on number of samples to convert (for testing)
    """
    
    if not HAS_CAIROSVG and not HAS_SVGLIB:
        print("❌ ERROR: No SVG conversion library available!")
        print("\nInstall one of:")
        print("  Option 1 (recommended): pip install cairosvg")
        print("  Option 2: pip install svglib reportlab")
        return
    
    converter = convert_svg_to_png_cairosvg if HAS_CAIROSVG else convert_svg_to_png_svglib
    converter_name = "cairosvg" if HAS_CAIROSVG else "svglib"
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create output directories
    images_dir = target_path / "images"
    annotations_dir = target_path / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using converter: {converter_name}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print("="*80)
    
    # Get all floor plan directories
    floor_plan_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    if max_samples:
        floor_plan_dirs = floor_plan_dirs[:max_samples]
        print(f"Converting {max_samples} samples (test mode)")
    else:
        print(f"Converting {len(floor_plan_dirs)} floor plans")
    
    successful = 0
    failed = 0
    
    for floor_dir in tqdm(floor_plan_dirs, desc="Converting"):
        floor_id = floor_dir.name
        
        try:
            # Copy image
            image_source = floor_dir / "F1_original.png"
            if image_source.exists():
                image_target = images_dir / f"{floor_id}.png"
                
                # Read image to get dimensions
                img = Image.open(image_source)
                width, height = img.size
                img.close()
                
                # Copy image
                import shutil
                shutil.copy2(image_source, image_target)
                
                # Convert SVG annotation to PNG
                svg_source = floor_dir / "model.svg"
                if svg_source.exists():
                    annotation_target = annotations_dir / f"{floor_id}.png"
                    
                    # Convert with same dimensions as image
                    if converter(svg_source, annotation_target, width, height):
                        successful += 1
                    else:
                        failed += 1
                        print(f"⚠️ Failed to convert: {floor_id}")
                else:
                    failed += 1
                    print(f"⚠️ No SVG found: {floor_id}")
            else:
                failed += 1
                print(f"⚠️ No image found: {floor_id}")
                
        except Exception as e:
            failed += 1
            print(f"❌ Error processing {floor_id}: {e}")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"✓ Successfully converted: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"\nDataset ready at: {target_path}")
    print(f"\nNext step:")
    print(f"  python src\\eda\\eda_analysis.py --dataset_path {target_path} --dataset_type cubicasa5k")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    print("="*80)
    
    if HAS_CAIROSVG:
        print("✓ cairosvg installed")
    else:
        print("✗ cairosvg NOT installed")
        print("  Install with: pip install cairosvg")
    
    if HAS_SVGLIB:
        print("✓ svglib installed")
    else:
        print("✗ svglib NOT installed")
        print("  Install with: pip install svglib reportlab")
    
    print("\n" + "="*80)
    
    if not HAS_CAIROSVG and not HAS_SVGLIB:
        print("❌ No SVG converter available!")
        print("\nREQUIRED ACTION:")
        print("  Install at least one SVG conversion library:")
        print("    pip install cairosvg")
        print("  OR")
        print("    pip install svglib reportlab")
        return False
    
    print("✓ Ready to convert SVG annotations")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CubiCasa5K SVG annotations to PNG")
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
        "--test",
        type=int,
        default=None,
        help="Convert only N samples for testing (e.g., --test 10)"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if dependencies are installed"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
    else:
        if check_dependencies():
            print("\nStarting conversion...")
            convert_cubicasa_svg_annotations(args.source, args.target, args.test)
        else:
            print("\n❌ Cannot proceed without SVG conversion library")
            sys.exit(1)
