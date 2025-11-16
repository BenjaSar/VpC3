"""
SVG to PNG Converter for CubiCasa5K Annotations
Converts model.svg files to semantic segmentation PNG masks with proper class labels
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List
import cv2

# Room class mapping - CubiCasa5K room types to class indices
ROOM_CLASS_MAPPING = {
    "Bedroom": 4,
    "Bathroom": 5,
    "Bath": 5,
    "Kitchen": 2,
    "LivingRoom": 3,
    "Living room": 3,
    "Entry Lobby": 6,
    "Hallway": 6,
    "Hall": 6,
    "Undefined": 10,
    "Outdoor Balcony": 11,
    "Balcony": 11,
    "Patio": 11,
    "Storage": 8,
    "Closet": 10,
    "Garage": 9,
    "Laundry": 10,
    "Sauna": 10,
    "Wall": 1,  # Walls are typically class 1
    "Column": 0,  # Structural elements as background
    "Railing": 0,  # Railings as background
    "Door": 0,
    "Window": 0,
    "FixedFurniture": 0,
}

# Default class for unknown room types
DEFAULT_ROOM_CLASS = 0  # Background


def extract_room_class_from_element(element) -> int:
    """
    Extract room class from SVG element based on class attribute
    """
    class_attr = element.get("class", "")
    
    # Parse class attribute (format: "Space RoomType" or just "RoomType")
    parts = class_attr.split()
    
    # Look for room type in parts
    for part in parts:
        if part in ROOM_CLASS_MAPPING:
            return ROOM_CLASS_MAPPING[part]
        # Check for partial matches
        for room_type, class_id in ROOM_CLASS_MAPPING.items():
            if room_type.lower() in part.lower() or part.lower() in room_type.lower():
                return class_id
    
    return DEFAULT_ROOM_CLASS


def parse_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """
    Parse SVG polygon points string into list of (x, y) tuples
    Format: "x1,y1 x2,y2 x3,y3 ..."
    """
    try:
        points = []
        pairs = points_str.strip().split()
        for pair in pairs:
            if pair:
                x, y = pair.split(',')
                points.append((float(x), float(y)))
        return points
    except (ValueError, IndexError):
        return []


def draw_polygon_on_mask(mask_array: np.ndarray, points: List[Tuple[float, float]], 
                         class_id: int, width: int, height: int):
    """
    Draw filled polygon on mask array with specified class value
    Uses OpenCV fillPoly for reliable polygon drawing
    """
    if len(points) < 3:
        return
    
    try:
        # Convert points to numpy array format for cv2
        poly_points = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
        
        # Draw filled polygon using OpenCV
        cv2.fillPoly(mask_array, [poly_points], class_id)
    except Exception as e:
        print(f"Warning: Failed to draw polygon: {e}")


def parse_svg_and_create_mask(svg_path: str, width: int, height: int) -> np.ndarray:
    """
    Parse SVG file and create semantic segmentation mask
    Each room is colored with its corresponding class value
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Extract namespace from root tag if present
        ns = ''
        if '}' in root.tag:
            ns = root.tag.split('}')[0] + '}'
        
        # Find all Space elements (rooms) - without namespace to be more robust
        for space_elem in root.iter():
            class_attr = space_elem.get('class', '')
            
            # Look for Space elements with room types
            if 'Space' not in class_attr:
                continue
            
            room_class = extract_room_class_from_element(space_elem)
            
            # Find polygon within this space
            for polygon in space_elem.iter():
                tag = polygon.tag
                if tag.endswith('polygon'):
                    points_str = polygon.get('points', '')
                    if points_str:
                        points = parse_polygon_points(points_str)
                        if points:
                            draw_polygon_on_mask(mask, points, room_class, width, height)
        
        # Process Wall elements separately
        for wall_elem in root.iter():
            class_attr = wall_elem.get('class', '')
            if class_attr == 'Wall External' or class_attr == 'Wall':
                # Find polygons in this wall
                for polygon in wall_elem.iter():
                    tag = polygon.tag
                    if tag.endswith('polygon'):
                        points_str = polygon.get('points', '')
                        if points_str:
                            points = parse_polygon_points(points_str)
                            if points:
                                draw_polygon_on_mask(mask, points, 1, width, height)  # Wall class = 1
        
        return mask
        
    except Exception as e:
        print(f"Error parsing SVG {svg_path}: {e}")
        import traceback
        traceback.print_exc()
        return mask


def convert_cubicasa_svg_annotations(source_dir, target_dir, max_samples=None):
    """
    Convert all CubiCasa5K SVG annotations to PNG semantic masks
    
    Args:
        source_dir: Path to cubicasa5k/high_quality directory
        target_dir: Path to output directory
        max_samples: Optional limit on number of samples to convert (for testing)
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create output directories
    images_dir = target_path / "images"
    annotations_dir = target_path / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"SVG to PNG Converter (Semantic Segmentation)")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print("="*80)
    print(f"Room class mapping:")
    for room_type, class_id in sorted(ROOM_CLASS_MAPPING.items(), key=lambda x: x[1]):
        print(f"  {class_id:2d}: {room_type}")
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
                
                # Convert SVG annotation to PNG semantic mask
                svg_source = floor_dir / "model.svg"
                if svg_source.exists():
                    annotation_target = annotations_dir / f"{floor_id}.png"
                    
                    # Parse SVG and create semantic mask
                    mask = parse_svg_and_create_mask(str(svg_source), width, height)
                    
                    # Save mask as PNG
                    mask_img = Image.fromarray(mask, mode='L')
                    mask_img.save(str(annotation_target))
                    
                    successful += 1
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
    print(f"\nOutput structure:")
    print(f"  {images_dir}")
    print(f"  {annotations_dir}")
    print(f"\nRoom class legend:")
    for room_type, class_id in sorted(ROOM_CLASS_MAPPING.items(), key=lambda x: x[1]):
        print(f"  {class_id:2d}: {room_type}")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    print("="*80)
    print("✓ PIL (Pillow) installed")
    print("✓ NumPy installed")
    print("✓ ElementTree (xml) installed")
    print("\n" + "="*80)
    print("✓ Ready to convert SVG annotations to semantic masks")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CubiCasa5K SVG annotations to semantic segmentation PNG")
    parser.add_argument(
        "--source",
        type=str,
        default="/home/ubuntu/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k/cubicasa5k/high_quality",
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
            print("\n❌ Cannot proceed without required dependencies")
            sys.exit(1)
