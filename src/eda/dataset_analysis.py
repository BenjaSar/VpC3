"""
Exploratory Data Analysis Module for Floor Plan Dataset
Handles dataset loading, statistics, and quality checks
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """
    Comprehensive dataset analysis for floor plan images and annotations
    """
    
    def __init__(self, dataset_path: str, dataset_type: str = "cubicasa5k"):
        """
        Initialize the dataset analyzer
        
        Args:
            dataset_path: Path to the dataset root directory
            dataset_type: Type of dataset ('cubicasa5k' or 'roboflow')
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        self.images_path = self.dataset_path / "images"
        self.annotations_path = self.dataset_path / "annotations"
        
        # Statistics storage
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'image_stats': {},
            'class_distribution': {},
            'quality_report': {}
        }
        
        logger.info(f"Initialized DatasetAnalyzer for {dataset_type}")
    
    def check_dataset_structure(self) -> Dict:
        """
        Verify dataset directory structure and file organization
        
        Returns:
            Dictionary with structure validation results
        """
        logger.info("Checking dataset structure...")
        
        structure_report = {
            'images_path_exists': self.images_path.exists(),
            'annotations_path_exists': self.annotations_path.exists(),
            'num_images': 0,
            'num_annotations': 0,
            'image_formats': [],
            'annotation_formats': [],
            'structure_valid': False
        }
        
        if self.images_path.exists():
            image_files = list(self.images_path.glob('*.*'))
            structure_report['num_images'] = len(image_files)
            structure_report['image_formats'] = list(set([f.suffix for f in image_files]))
            
        if self.annotations_path.exists():
            annotation_files = list(self.annotations_path.glob('*.*'))
            structure_report['num_annotations'] = len(annotation_files)
            structure_report['annotation_formats'] = list(set([f.suffix for f in annotation_files]))
        
        structure_report['structure_valid'] = (
            structure_report['images_path_exists'] and 
            structure_report['annotations_path_exists']
        )
        
        logger.info(f"Dataset structure check: {structure_report['structure_valid']}")
        logger.info(f"Found {structure_report['num_images']} images")
        logger.info(f"Found {structure_report['num_annotations']} annotations")
        
        return structure_report
    
    def analyze_image_properties(self) -> pd.DataFrame:
        """
        Analyze properties of all images in the dataset
        
        Returns:
            DataFrame with image statistics
        """
        logger.info("Analyzing image properties...")
        
        image_stats = []
        image_files = sorted(list(self.images_path.glob('*.*')))
        
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Load image
                img = cv2.imread(str(img_path))
                
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                height, width = img.shape[:2]
                file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
                
                image_stats.append({
                    'filename': img_path.name,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height if height > 0 else 0,
                    'file_size_mb': file_size,
                    'resolution': f"{width}x{height}",
                    'channels': img.shape[2] if len(img.shape) > 2 else 1
                })
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
        
        df_stats = pd.DataFrame(image_stats)
        
        if len(df_stats) > 0:
            self.stats['image_stats'] = {
                'total_images': len(df_stats),
                'min_width': int(df_stats['width'].min()),
                'max_width': int(df_stats['width'].max()),
                'mean_width': float(df_stats['width'].mean()),
                'std_width': float(df_stats['width'].std()),
                'min_height': int(df_stats['height'].min()),
                'max_height': int(df_stats['height'].max()),
                'mean_height': float(df_stats['height'].mean()),
                'std_height': float(df_stats['height'].std()),
                'aspect_ratio_stats': {
                    'min': float(df_stats['aspect_ratio'].min()),
                    'max': float(df_stats['aspect_ratio'].max()),
                    'mean': float(df_stats['aspect_ratio'].mean()),
                    'std': float(df_stats['aspect_ratio'].std())
                },
                'total_storage_gb': float(df_stats['file_size_mb'].sum() / 1024),
                'resolution_distribution': df_stats['resolution'].value_counts().to_dict()
            }
            
            logger.info(f"Analyzed {len(df_stats)} images")
            logger.info(f"Image dimensions: {self.stats['image_stats']['min_width']}x{self.stats['image_stats']['min_height']} to "
                       f"{self.stats['image_stats']['max_width']}x{self.stats['image_stats']['max_height']}")
        
        return df_stats
    
    def analyze_annotations(self, mask_suffix: str = ".png") -> Dict:
        """
        Analyze segmentation mask annotations
        
        Args:
            mask_suffix: File extension for mask files
            
        Returns:
            Dictionary with annotation statistics
        """
        logger.info("Analyzing annotations...")
        
        annotation_analysis = {
            'total_masks': 0,
            'class_distribution': {},
            'pixel_statistics': {},
            'missing_annotations': [],
            'mask_dimensions': {}
        }
        
        mask_files = sorted(list(self.annotations_path.glob(f'*{mask_suffix}')))
        annotation_analysis['total_masks'] = len(mask_files)
        
        class_pixel_counts = Counter()
        class_image_counts = Counter()
        pixel_stats = {
            'total_pixels': 0,
            'valid_pixels': 0,
            'background_pixels': 0,
            'annotated_pixels': 0
        }
        
        mask_dimensions = []
        
        for mask_path in tqdm(mask_files, desc="Processing masks"):
            try:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if mask is None:
                    logger.warning(f"Failed to load mask: {mask_path}")
                    annotation_analysis['missing_annotations'].append(str(mask_path))
                    continue
                
                # Count pixels per class
                unique_classes = np.unique(mask)
                for class_id in unique_classes:
                    pixel_count = np.sum(mask == class_id)
                    class_pixel_counts[class_id] += pixel_count
                    if pixel_count > 0:
                        class_image_counts[class_id] += 1
                
                pixel_stats['total_pixels'] += mask.size
                pixel_stats['valid_pixels'] += np.sum(mask != 255)  # 255 is typically background
                pixel_stats['background_pixels'] += np.sum(mask == 0)
                pixel_stats['annotated_pixels'] += np.sum(mask != 0)
                
                mask_dimensions.append({
                    'filename': mask_path.name,
                    'width': mask.shape[1],
                    'height': mask.shape[0]
                })
                
            except Exception as e:
                logger.error(f"Error processing mask {mask_path}: {str(e)}")
        
        annotation_analysis['class_distribution'] = {
            'pixel_counts': dict(class_pixel_counts),
            'image_counts': dict(class_image_counts),
            'num_classes': len(class_pixel_counts)
        }
        annotation_analysis['pixel_statistics'] = pixel_stats
        annotation_analysis['mask_dimensions'] = {
            'total_masks': len(mask_dimensions),
            'widths': [m['width'] for m in mask_dimensions],
            'heights': [m['height'] for m in mask_dimensions]
        }
        
        self.stats['class_distribution'] = annotation_analysis['class_distribution']
        
        logger.info(f"Analyzed {len(mask_files)} annotations")
        logger.info(f"Found {annotation_analysis['class_distribution']['num_classes']} unique classes")
        
        return annotation_analysis
    
    def analyze_class_imbalance(self, annotation_analysis: Dict) -> Dict:
        """
        Analyze class imbalance in the dataset
        
        Args:
            annotation_analysis: Results from analyze_annotations()
            
        Returns:
            Dictionary with class imbalance metrics
        """
        logger.info("Analyzing class imbalance...")
        
        pixel_counts = annotation_analysis['class_distribution']['pixel_counts']
        image_counts = annotation_analysis['class_distribution']['image_counts']
        
        imbalance_analysis = {
            'class_weights': {},
            'imbalance_ratio': 0.0,
            'most_common_class': None,
            'least_common_class': None,
            'class_statistics': {}
        }
        
        if pixel_counts:
            total_pixels = sum(pixel_counts.values())
            max_pixels = max(pixel_counts.values())
            min_pixels = min(pixel_counts.values())
            
            # Calculate class weights (inverse frequency)
            for class_id, count in pixel_counts.items():
                weight = total_pixels / (len(pixel_counts) * max(count, 1))
                imbalance_analysis['class_weights'][class_id] = float(weight)
                
                imbalance_analysis['class_statistics'][class_id] = {
                    'pixel_count': int(count),
                    'percentage': float((count / total_pixels) * 100),
                    'image_count': int(image_counts.get(class_id, 0)),
                    'weight': float(weight)
                }
            
            imbalance_analysis['imbalance_ratio'] = float(max_pixels / max(min_pixels, 1))
            imbalance_analysis['most_common_class'] = max(pixel_counts, key=pixel_counts.get)
            imbalance_analysis['least_common_class'] = min(pixel_counts, key=pixel_counts.get)
            
            logger.info(f"Class imbalance ratio: {imbalance_analysis['imbalance_ratio']:.2f}:1")
        
        return imbalance_analysis
    
    def generate_quality_report(self) -> Dict:
        """
        Generate comprehensive data quality report
        
        Returns:
            Dictionary with quality assessment
        """
        logger.info("Generating quality report...")
        
        report = {
            'data_completeness': 0.0,
            'image_annotation_match': 0.0,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': [],
            'summary': {}
        }
        
        image_files = list(self.images_path.glob('*.*'))
        mask_files = list(self.annotations_path.glob('*.png'))
        
        # Check completeness
        if len(image_files) > 0:
            report['data_completeness'] = len(mask_files) / len(image_files)
        
        # Check image-annotation matching
        image_basenames = {f.stem for f in image_files}
        mask_basenames = {f.stem for f in mask_files}
        
        matched = len(image_basenames & mask_basenames)
        if len(image_files) > 0:
            report['image_annotation_match'] = matched / len(image_files)
        
        # Overall quality score
        report['quality_score'] = (
            report['data_completeness'] * 0.5 + 
            report['image_annotation_match'] * 0.5
        )
        
        # Issues identification
        if report['data_completeness'] < 0.95:
            report['issues'].append(f"Only {report['data_completeness']:.1%} of images have annotations")
            report['recommendations'].append("Check for missing annotation files")
        
        if report['image_annotation_match'] < 0.95:
            report['issues'].append(f"Only {report['image_annotation_match']:.1%} image-annotation pairs match")
            report['recommendations'].append("Verify naming conventions match between images and annotations")
        
        if report['quality_score'] >= 0.95:
            report['summary']['status'] = "EXCELLENT"
        elif report['quality_score'] >= 0.85:
            report['summary']['status'] = "GOOD"
        elif report['quality_score'] >= 0.75:
            report['summary']['status'] = "ACCEPTABLE"
        else:
            report['summary']['status'] = "POOR"
        
        self.stats['quality_report'] = report
        logger.info(f"Quality score: {report['quality_score']:.2%} - {report['summary']['status']}")
        
        return report
    
    def get_summary_statistics(self) -> Dict:
        """
        Get comprehensive summary statistics
        
        Returns:
            Dictionary with all statistics
        """
        return self.stats
    
    def export_statistics(self, output_path: str):
        """
        Export statistics to JSON file
        
        Args:
            output_path: Path to save statistics JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            """Recursively convert numpy types to Python native types"""
            import numpy as np
            
            if isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        stats_converted = convert_types(self.stats)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_converted, f, indent=4, default=str)
        logger.info(f"Statistics exported to {output_path}")
