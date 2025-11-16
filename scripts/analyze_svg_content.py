#!/usr/bin/env python3
"""
SVG Content Analyzer: Investigate Room Type Distribution
Identifies why certain classes are dominating the dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np

# Room class mapping from svg_to_png_converter
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
    "Wall": 1,
    "Column": 0,
    "Railing": 0,
    "Door": 0,
    "Window": 0,
    "FixedFurniture": 0,
}

def extract_room_type_from_class_attr(class_attr):
    """Extract room type from SVG class attribute"""
    parts = class_attr.split()
    
    for part in parts:
        if part in ROOM_CLASS_MAPPING:
            return part
        for room_type in ROOM_CLASS_MAPPING.keys():
            if room_type.lower() in part.lower() or part.lower() in room_type.lower():
                return room_type
    
    return "UNMAPPED"

def analyze_svg_files():
    """Analyze all SVG files in the dataset"""
    
    print("=" * 80)
    print("SVG CONTENT ANALYZER")
    print("=" * 80)
    
    # Find SVG files - try original location first
    svg_dir = Path("/home/ubuntu/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k/cubicasa5k/high_quality")
    
    if not svg_dir.exists():
        print(f"❌ SVG directory not found: {svg_dir}")
        print("   Make sure the CubiCasa5K dataset is downloaded from Kaggle")
        return
    
    print(f"✓ Using SVG directory: {svg_dir}")
    
    # Find all floor plan directories with SVG files
    svg_files = []
    try:
        for floor_dir in sorted(svg_dir.iterdir()):
            if floor_dir.is_dir():
                svg_file = floor_dir / "model.svg"
                if svg_file.exists():
                    svg_files.append((floor_dir.name, svg_file))
    except Exception as e:
        print(f"❌ Error reading SVG directory: {e}")
        return
    
    print(f"\n✓ Found {len(svg_files)} SVG files")
    
    if len(svg_files) == 0:
        print("❌ No SVG files found in the directory!")
        print("   Checking directory contents...")
        try:
            contents = list(svg_dir.iterdir())[:5]
            for item in contents:
                print(f"   - {item.name} ({'dir' if item.is_dir() else 'file'})")
        except:
            pass
        return
    
    # Analyze room types across all files
    room_type_counter = Counter()
    all_elements_counter = Counter()
    room_type_to_class = {}
    
    print(f"Analyzing SVG content (sampling first 100 SVGs)...")
    
    for i, (floor_id, svg_file) in enumerate(svg_files[:100]):
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            
            # First, sample what elements exist
            if i < 3:  # Show first 3 files
                print(f"\n  Sample SVG #{i+1}: {floor_id}")
                for elem in root.iter():
                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    class_attr = elem.get('class', '')
                    if class_attr:
                        all_elements_counter[f"{tag}:{class_attr}"] += 1
                        if i < 1:  # Print for first file only
                            print(f"    - {tag} → class='{class_attr}'")
            
            # Count Space elements (rooms)
            for elem in root.iter():
                class_attr = elem.get('class', '')
                
                # Look for Space elements
                if 'Space' in class_attr:
                    room_type = extract_room_type_from_class_attr(class_attr)
                    room_type_counter[room_type] += 1
                    
                    if room_type != "UNMAPPED":
                        class_id = ROOM_CLASS_MAPPING.get(room_type, 0)
                        room_type_to_class[room_type] = class_id
                
                # Count Wall elements
                if class_attr == 'Wall' or class_attr == 'Wall External':
                    room_type_counter['Wall'] += 1
                    room_type_to_class['Wall'] = 1
        
        except Exception as e:
            print(f"  ⚠️  Error parsing {floor_id}: {e}")
    
    # Print room type distribution
    print("\n" + "=" * 80)
    print("ROOM TYPE DISTRIBUTION (Sample of 100 SVGs)")
    print("=" * 80)
    
    total_rooms = sum(room_type_counter.values())
    
    # Sort by frequency
    for room_type, count in room_type_counter.most_common():
        pct = (count / total_rooms * 100) if total_rooms > 0 else 0
        class_id = ROOM_CLASS_MAPPING.get(room_type, 0)
        bar = "█" * int(pct / 2)
        print(f"  {room_type:20s} → Class {class_id:2d}: {count:5d} ({pct:5.2f}%) {bar}")
    
    # Print class distribution based on room types
    print("\n" + "=" * 80)
    print("ROOM TYPES MAPPED TO CLASSES")
    print("=" * 80)
    
    class_room_mapping = {}
    for room_type, class_id in sorted(room_type_to_class.items(), key=lambda x: x[1]):
        if class_id not in class_room_mapping:
            class_room_mapping[class_id] = []
        class_room_mapping[class_id].append(room_type)
    
    for class_id in sorted(class_room_mapping.keys()):
        room_types = class_room_mapping[class_id]
        print(f"  Class {class_id:2d}: {', '.join(room_types)}")
    
    # Identify the issue
    print("\n" + "=" * 80)
    print("ANALYSIS & DIAGNOSIS")
    print("=" * 80)
    
    unmapped_count = room_type_counter.get('UNMAPPED', 0)
    total = sum(room_type_counter.values())
    
    print(f"\n✓ Total room elements found: {total}")
    
    if total == 0:
        print("\n⚠️  WARNING: No room elements found in SVG files!")
        print("   This might mean:")
        print("   - SVG structure doesn't have 'Space' class elements")
        print("   - Different room type encoding in original SVGs")
        print("\n   Showing all unique element types found:")
        for elem_type, count in all_elements_counter.most_common(20):
            print(f"   - {elem_type}: {count}")
        return
    
    print(f"✓ Unmapped room types: {unmapped_count} ({unmapped_count/total*100:.2f}%)")
    
    if unmapped_count > total * 0.1:
        print(f"\n⚠️  CRITICAL: {unmapped_count/total*100:.2f}% of rooms are unmapped!")
        print("   This is likely the source of class imbalance.")
        print("   Many undefined rooms are being assigned to default/storage classes.")
    
    # Check for dominant classes
    print("\n" + "=" * 80)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 80)
    
    class_counts = Counter()
    for room_type, count in room_type_counter.items():
        if room_type != "UNMAPPED":
            class_id = ROOM_CLASS_MAPPING.get(room_type, 0)
            class_counts[class_id] += count
    
    print("\nEstimated class distribution:")
    total_class_pixels = sum(class_counts.values())
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        pct = (count / total_class_pixels * 100)
        bar = "█" * int(pct / 5)
        print(f"  Class {class_id:2d}: {count:5d} ({pct:5.2f}%) {bar}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    dominant_classes = [c for c in class_counts if class_counts[c] > total_class_pixels * 0.3]
    
    if len(dominant_classes) >= 2:
        print("\n⚠️  EXTREME CLASS IMBALANCE DETECTED!")
        print(f"   {len(dominant_classes)} classes account for >{30}% each")
        print("\n   Solutions (in order of effectiveness):")
        print("   1. Implement Focal Loss (best for extreme imbalance)")
        print("   2. Use aggressive class weights (weight = 1/frequency)")
        print("   3. Oversample minority classes or undersample majority")
        print("   4. Use stratified k-fold training")
        print("   5. Try weighted focal loss combination")
    
    print("\n   Implementation:")
    print("   - Replace CrossEntropyLoss with FocalLoss in train.py")
    print("   - Adjust hyperparameters (alpha, gamma)")
    print("   - Consider data rebalancing strategies")

if __name__ == "__main__":
    analyze_svg_files()
