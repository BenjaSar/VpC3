#!/usr/bin/env python3
"""
Fix mask class values from 256 classes to 34 CubiCasa5K classes
CRITICAL: Your masks have values 0-255 but model expects 0-33
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# CubiCasa5K 256 -> 34 class mapping
# Based on CubiCasa5K dataset structure
CUBICASA_CLASS_MAPPING = {
    0: 0,      # Background/Outdoor → Background
    255: 0,    # White/Unknown → Background  
    1: 1,      # Wall
    2: 2,      # Kitchen
    3: 3,      # Living room
    4: 4,      # Bedroom
    5: 5,      # Bath
    6: 6,      # Entry/Hall
    7: 7,      # Railing
    8: 8,      # Storage
    9: 9,      # Garage
    10: 10,    # Undefined
    11: 11,    # Interior door
    12: 12,    # Exterior door
    13: 13,    # Window
    # Add more mappings based on your specific dataset
    # For now, map everything else to class 10 (undefined)
}

def create_mapping_table():
    """Create a lookup table for fast remapping"""
    # Default: map everything to undefined (class 10)
    mapping_table = np.full(256, 10, dtype=np.uint8)
    
    # Apply known mappings
    for old_class, new_class in CUBICASA_CLASS_MAPPING.items():
        mapping_table[old_class] = new_class
    
    # Map common values
    mapping_table[0] = 0      # Background
    mapping_table[255] = 0    # Background/White
    
    # If you have specific class info, update this
    # For example: walls (values 1-5) → class 1
    for i in range(1, 6):
        mapping_table[i] = 1  # Wall
    
    # Rooms (values 6-20) → different room types
    for i in range(6, 15):
        if i <= 9:
            mapping_table[i] = min(i - 5, 5)  # Map to room classes 1-4
        else:
            mapping_table[i] = 10  # Undefined
    
    return mapping_table


def remap_mask(mask, mapping_table):
    """Remap mask values using lookup table"""
    return mapping_table[mask]


def analyze_current_masks(masks_dir):
    """Analyze what class values actually exist in masks"""
    print("Analyzing mask values...")
    masks_dir = Path(masks_dir)
    
    all_values = set()
    value_counts = {}
    
    for mask_file in tqdm(list(masks_dir.glob("*.png"))[:100], desc="Scanning masks"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        unique = np.unique(mask)
        all_values.update(unique)
        
        for val in unique:
            value_counts[int(val)] = value_counts.get(int(val), 0) + 1
    
    print(f"\nFound {len(all_values)} unique class values:")
    print(f"Range: {min(all_values)} to {max(all_values)}")
    print(f"\nMost common values:")
    sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    for val, count in sorted_counts[:20]:
        print(f"  Value {val}: appears in {count} masks")
    
    return all_values, value_counts


def fix_masks(input_dir, output_dir, mapping_table):
    """Remap all masks from 256 classes to 34 classes"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_files = list(input_dir.glob("*.png"))
    print(f"\nRemapping {len(mask_files)} masks...")
    
    stats = {
        'total': len(mask_files),
        'processed': 0,
        'errors': 0,
        'class_distribution': {}
    }
    
    for mask_file in tqdm(mask_files, desc="Remapping masks"):
        try:
            # Read mask
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Remap to 34 classes
            remapped_mask = remap_mask(mask, mapping_table)
            
            # Count classes
            unique = np.unique(remapped_mask)
            for cls in unique:
                stats['class_distribution'][int(cls)] = stats['class_distribution'].get(int(cls), 0) + 1
            
            # Save
            output_path = output_dir / mask_file.name
            cv2.imwrite(str(output_path), remapped_mask)
            stats['processed'] += 1
            
        except Exception as e:
            print(f"Error processing {mask_file.name}: {e}")
            stats['errors'] += 1
    
    return stats


def main():
    print("="*80)
    print("FIX MASK CLASSES: 256 → 34")
    print("="*80)
    
    # Paths
    INPUT_MASKS_DIR = 'data/processed/annotations'
    OUTPUT_MASKS_DIR = 'data/processed_fixed/annotations'
    INPUT_IMAGES_DIR = 'data/processed/images'
    OUTPUT_IMAGES_DIR = 'data/processed_fixed/images'
    
    # Step 1: Analyze current masks
    print("\nStep 1: Analyzing current mask values...")
    all_values, value_counts = analyze_current_masks(INPUT_MASKS_DIR)
    
    if max(all_values) <= 33:
        print("\n✓ Masks already have valid range [0-33]. No remapping needed!")
        return
    
    print(f"\n⚠️  WARNING: Masks have values up to {max(all_values)}, but model expects [0-33]")
    print("This MUST be fixed before training!")
    
    # Step 2: Create mapping
    print("\nStep 2: Creating class mapping...")
    mapping_table = create_mapping_table()
    
    print("\nMapping table (first 20 values):")
    for i in range(min(20, len(mapping_table))):
        print(f"  {i} → {mapping_table[i]}")
    print(f"  ...")
    print(f"  255 → {mapping_table[255]}")
    
    # Save mapping
    mapping_dict = {int(i): int(mapping_table[i]) for i in range(256)}
    with open('class_mapping_256_to_34.json', 'w') as f:
        json.dump(mapping_dict, f, indent=2)
    print("\n✓ Mapping saved to: class_mapping_256_to_34.json")
    
    # Step 3: Remap masks
    print("\nStep 3: Remapping masks...")
    stats = fix_masks(INPUT_MASKS_DIR, OUTPUT_MASKS_DIR, mapping_table)
    
    # Step 4: Copy images (unchanged)
    print("\nStep 4: Copying images...")
    import shutil
    Path(OUTPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    
    images = list(Path(INPUT_IMAGES_DIR).glob("*.png"))
    for img in tqdm(images, desc="Copying images"):
        shutil.copy(img, Path(OUTPUT_IMAGES_DIR) / img.name)
    
    # Report
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total masks: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Errors: {stats['errors']}")
    
    print("\nNew class distribution (in masks):")
    for cls in sorted(stats['class_distribution'].keys()):
        count = stats['class_distribution'][cls]
        print(f"  Class {cls}: {count} masks")
    
    print(f"\n✓ Fixed data saved to: {OUTPUT_MASKS_DIR}")
    print(f"✓ Images copied to: {OUTPUT_IMAGES_DIR}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Update train.py and train_fixed.py to use the new paths:")
    print(f"   'images_dir': '{OUTPUT_IMAGES_DIR}'")
    print(f"   'masks_dir': '{OUTPUT_MASKS_DIR}'")
    print("\n2. Run diagnosis again:")
    print("   python diagnose_model.py")
    print("\n3. Train with fixed data:")
    print("   python train_fixed.py")
    
    print("\n⚠️  IMPORTANT: The class mapping used is a GUESS!")
    print("You should verify the mapping matches your dataset's class definitions.")
    print("Check the CubiCasa5K documentation for the correct mapping.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
