"""
Main EDA Analysis Script
Orchestrates the complete exploratory data analysis pipeline
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eda.dataset_analysis import DatasetAnalyzer
from src.eda.visualization import VisualizationEngine
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def print_separator(title: str = "", width: int = 80):
    """Print formatted separator"""
    if title:
        logger.info(f"\n{'=' * width}")
        logger.info(f"{title.center(width)}")
        logger.info(f"{'=' * width}\n")
    else:
        logger.info("=" * width)


def print_report(title: str, report: dict, indent: int = 2):
    """Print formatted report"""
    print_separator(title)
    for key, value in report.items():
        if isinstance(value, dict):
            logger.info(f"{' ' * indent}{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"{' ' * (indent * 2)}{sub_key}: {sub_value}")
        elif isinstance(value, list):
            logger.info(f"{' ' * indent}{key}:")
            for item in value:
                logger.info(f"{' ' * (indent * 2)}- {item}")
        else:
            logger.info(f"{' ' * indent}{key}: {value}")


def print_dataframe_summary(title: str, df):
    """Print DataFrame summary"""
    print_separator(title)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nFirst few rows:")
    logger.info(f"\n{df.head()}")
    logger.info(f"\nData types:")
    logger.info(f"\n{df.dtypes}")
    logger.info(f"\nStatistical summary:")
    logger.info(f"\n{df.describe()}")


def run_eda_analysis(dataset_path: str, output_dir: str = None, dataset_type: str = "cubicasa5k"):
    """
    Run complete EDA analysis on floor plan dataset
    
    Args:
        dataset_path: Path to dataset root directory
        output_dir: Output directory for visualizations and reports
        dataset_type: Type of dataset ('cubicasa5k' or 'roboflow')
    """
    
    try:
        print_separator("FLOOR PLAN DATASET EXPLORATORY DATA ANALYSIS (EDA)")
        logger.info(f"Dataset Type: {dataset_type}")
        logger.info(f"Dataset Path: {dataset_path}")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./eda_output_{timestamp}"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output Directory: {output_dir}\n")
        
        # Initialize analyzer and visualization engine
        analyzer = DatasetAnalyzer(dataset_path, dataset_type)
        viz_engine = VisualizationEngine(output_dir)
        
        # =====================================================================
        # STEP 1: Check Dataset Structure
        # =====================================================================
        print_separator("STEP 1: Dataset Structure Validation")
        structure_report = analyzer.check_dataset_structure()
        print_report("Dataset Structure Report", structure_report)
        
        if not structure_report['structure_valid']:
            logger.error("Dataset structure is invalid! Please check directory organization.")
            logger.error("Expected structure:")
            logger.error("  dataset_root/")
            logger.error("    ├── images/")
            logger.error("    └── annotations/")
            return None
        
        # =====================================================================
        # STEP 2: Analyze Image Properties
        # =====================================================================
        print_separator("STEP 2: Image Properties Analysis")
        logger.info(f"Analyzing {structure_report['num_images']} images...\n")
        
        df_stats = analyzer.analyze_image_properties()
        
        if len(df_stats) == 0:
            logger.error("No images were successfully processed!")
            return None
        
        print_dataframe_summary("Image Statistics", df_stats)
        
        # Save image stats CSV
        csv_path = Path(output_dir) / "image_statistics.csv"
        df_stats.to_csv(csv_path, index=False)
        logger.info(f"Image statistics saved to: {csv_path}")
        
        # =====================================================================
        # STEP 3: Analyze Annotations
        # =====================================================================
        print_separator("STEP 3: Annotation Analysis")
        logger.info(f"Analyzing {structure_report['num_annotations']} annotations...\n")
        
        annotation_analysis = analyzer.analyze_annotations()
        print_report("Annotation Analysis Report", annotation_analysis)
        
        # =====================================================================
        # STEP 4: Class Imbalance Analysis
        # =====================================================================
        print_separator("STEP 4: Class Imbalance Analysis")
        
        imbalance_analysis = analyzer.analyze_class_imbalance(annotation_analysis)
        
        logger.info(f"Number of Classes: {len(imbalance_analysis['class_statistics'])}")
        logger.info(f"Imbalance Ratio: {imbalance_analysis['imbalance_ratio']:.2f}:1")
        logger.info(f"Most Common Class: {imbalance_analysis['most_common_class']}")
        logger.info(f"Least Common Class: {imbalance_analysis['least_common_class']}")
        
        logger.info("\nClass-wise Statistics:")
        for class_id in sorted(imbalance_analysis['class_statistics'].keys()):
            stats = imbalance_analysis['class_statistics'][class_id]
            logger.info(f"\n  Class {class_id}:")
            logger.info(f"    - Pixel Count: {stats['pixel_count']:,}")
            logger.info(f"    - Percentage: {stats['percentage']:.2f}%")
            logger.info(f"    - Image Count: {stats['image_count']}")
            logger.info(f"    - Weight: {stats['weight']:.4f}")
        
        # =====================================================================
        # STEP 5: Data Quality Assessment
        # =====================================================================
        print_separator("STEP 5: Data Quality Assessment")
        
        quality_report = analyzer.generate_quality_report()
        print_report("Data Quality Report", quality_report)
        
        # =====================================================================
        # STEP 6: Generate Visualizations
        # =====================================================================
        print_separator("STEP 6: Generating Visualizations")
        
        logger.info("Creating visualization plots...")
        viz_engine.create_eda_report(
            df_stats=df_stats,
            annotation_analysis=annotation_analysis,
            imbalance_analysis=imbalance_analysis,
            quality_report=quality_report,
            images_path=str(analyzer.images_path),
            annotations_path=str(analyzer.annotations_path)
        )
        
        logger.info(f"\nVisualizations saved to: {output_dir}")
        
        # =====================================================================
        # STEP 7: Export Summary Report
        # =====================================================================
        print_separator("STEP 7: Exporting Summary Report")
        
        # Get all statistics
        all_stats = analyzer.get_summary_statistics()
        
        # Add additional computed statistics
        all_stats['analysis_summary'] = {
            'dataset_type': dataset_type,
            'analysis_date': datetime.now().isoformat(),
            'total_images': len(df_stats),
            'total_annotations': annotation_analysis['total_masks'],
            'num_classes': annotation_analysis['class_distribution']['num_classes'],
            'imbalance_ratio': imbalance_analysis['imbalance_ratio'],
            'quality_status': quality_report['summary']['status'],
            'quality_score': quality_report['quality_score'],
            'data_completeness': quality_report['data_completeness'],
            'image_annotation_match': quality_report['image_annotation_match']
        }
        
        # Save JSON report
        json_path = Path(output_dir) / "eda_report.json"
        analyzer.export_statistics(str(json_path))
        logger.info(f"EDA report exported to: {json_path}")
        
        # Create human-readable text report
        text_report_path = Path(output_dir) / "EDA_REPORT.txt"
        create_text_report(all_stats, text_report_path)
        logger.info(f"Text report saved to: {text_report_path}")
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print_separator("EDA ANALYSIS COMPLETED SUCCESSFULLY")
        
        logger.info("\nGenerated Files:")
        logger.info(f"  ✓ image_statistics.csv")
        logger.info(f"  ✓ eda_report.json")
        logger.info(f"  ✓ EDA_REPORT.txt")
        logger.info(f"  ✓ 01_image_dimensions.png")
        logger.info(f"  ✓ 02_file_size_distribution.png")
        logger.info(f"  ✓ 03_class_distribution.png")
        logger.info(f"  ✓ 04_pixel_statistics.png")
        logger.info(f"  ✓ 05_mask_dimensions.png")
        logger.info(f"  ✓ 06_quality_report.png")
        logger.info(f"  ✓ 07_sample_floor_plans.png")
        
        logger.info(f"\nOutput Directory: {output_dir}")
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'success': True,
            'output_dir': output_dir,
            'stats': all_stats,
            'df_stats': df_stats
        }
        
    except Exception as e:
        logger.error(f"Error during EDA analysis: {str(e)}", exc_info=True)
        return None


def create_text_report(stats: dict, output_path: Path):
    """
    Create human-readable text report
    
    Args:
        stats: All statistics dictionary
        output_path: Path to save text report
    """
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("FLOOR PLAN DATASET EXPLORATORY DATA ANALYSIS (EDA) REPORT".center(80))
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary Section
    if 'analysis_summary' in stats:
        summary = stats['analysis_summary']
        report_lines.append("ANALYSIS SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Analysis Date: {summary['analysis_date']}")
        report_lines.append(f"Dataset Type: {summary['dataset_type']}")
        report_lines.append(f"Total Images: {summary['total_images']:,}")
        report_lines.append(f"Total Annotations: {summary['total_annotations']:,}")
        report_lines.append(f"Number of Classes: {summary['num_classes']}")
        report_lines.append(f"Class Imbalance Ratio: {summary['imbalance_ratio']:.2f}:1")
        report_lines.append(f"Quality Status: {summary['quality_status']}")
        report_lines.append(f"Quality Score: {summary['quality_score']:.2%}")
        report_lines.append(f"Data Completeness: {summary['data_completeness']:.2%}")
        report_lines.append(f"Image-Annotation Match: {summary['image_annotation_match']:.2%}")
        report_lines.append("")
    
    # Image Statistics Section
    if 'image_stats' in stats:
        img_stats = stats['image_stats']
        report_lines.append("IMAGE STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Images: {img_stats['total_images']}")
        report_lines.append(f"Width Range: {img_stats['min_width']} - {img_stats['max_width']} pixels")
        report_lines.append(f"Width Mean: {img_stats['mean_width']:.2f} ± {img_stats['std_width']:.2f} pixels")
        report_lines.append(f"Height Range: {img_stats['min_height']} - {img_stats['max_height']} pixels")
        report_lines.append(f"Height Mean: {img_stats['mean_height']:.2f} ± {img_stats['std_height']:.2f} pixels")
        report_lines.append(f"Aspect Ratio Range: {img_stats['aspect_ratio_stats']['min']:.2f} - {img_stats['aspect_ratio_stats']['max']:.2f}")
        report_lines.append(f"Aspect Ratio Mean: {img_stats['aspect_ratio_stats']['mean']:.2f} ± {img_stats['aspect_ratio_stats']['std']:.2f}")
        report_lines.append(f"Total Storage: {img_stats['total_storage_gb']:.2f} GB")
        report_lines.append("")
    
    # Class Distribution Section
    if 'class_distribution' in stats:
        class_dist = stats['class_distribution']
        report_lines.append("CLASS DISTRIBUTION")
        report_lines.append("-" * 80)
        report_lines.append(f"Number of Classes: {class_dist['num_classes']}")
        report_lines.append("")
        report_lines.append("Top 10 Most Common Classes:")
        
        # Sort by pixel count
        sorted_classes = sorted(class_dist['pixel_counts'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        for class_id, pixel_count in sorted_classes:
            total_pixels = sum(class_dist['pixel_counts'].values())
            percentage = (pixel_count / total_pixels) * 100
            image_count = class_dist['image_counts'].get(class_id, 0)
            report_lines.append(f"  Class {class_id}: {pixel_count:,} pixels ({percentage:.2f}%) in {image_count} images")
        
        report_lines.append("")
    
    # Quality Report Section
    if 'quality_report' in stats:
        qr = stats['quality_report']
        report_lines.append("DATA QUALITY ASSESSMENT")
        report_lines.append("-" * 80)
        report_lines.append(f"Status: {qr['summary']['status']}")
        report_lines.append(f"Quality Score: {qr['quality_score']:.2%}")
        report_lines.append(f"Data Completeness: {qr['data_completeness']:.2%}")
        report_lines.append(f"Image-Annotation Match: {qr['image_annotation_match']:.2%}")
        
        if qr['issues']:
            report_lines.append(f"\nIssues Found ({len(qr['issues'])}):")
            for issue in qr['issues']:
                report_lines.append(f"  ⚠ {issue}")
        
        if qr['recommendations']:
            report_lines.append(f"\nRecommendations ({len(qr['recommendations'])}):")
            for rec in qr['recommendations']:
                report_lines.append(f"  ➜ {rec}")
        
        report_lines.append("")
    
        # Recommendations Section
    report_lines.append("EDA RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("Based on the analysis, here are recommendations for model training:")
    report_lines.append("")
    
    if 'image_stats' in stats:
        img_stats = stats['image_stats']
        report_lines.append(f"1. INPUT IMAGE SIZE:")
        report_lines.append(f"   Recommended input size: {int(img_stats['mean_width']/32)*32}x{int(img_stats['mean_height']/32)*32}")
        report_lines.append(f"   (Divisible by 32 for patch-based models)")
        report_lines.append("")
    
    if 'analysis_summary' in stats:
        summary = stats['analysis_summary']
        report_lines.append(f"2. CLASS IMBALANCE HANDLING:")
        if summary['imbalance_ratio'] > 5:
            report_lines.append(f"   ⚠ High class imbalance detected ({summary['imbalance_ratio']:.1f}:1)")
            report_lines.append(f"   → Use weighted loss functions (Focal Loss, Weighted Cross-Entropy)")
            report_lines.append(f"   → Consider class balancing augmentation")
        else:
            report_lines.append(f"   ✓ Moderate class balance ({summary['imbalance_ratio']:.1f}:1)")
            report_lines.append(f"   → Standard loss functions should work well")
        report_lines.append("")
    
    report_lines.append("3. DATA AUGMENTATION:")
    report_lines.append("   Recommended augmentations for floor plans:")
    report_lines.append("   • Rotation (±15 degrees)")
    report_lines.append("   • Horizontal/Vertical Flips")
    report_lines.append("   • Elastic Deformation")
    report_lines.append("   • Random Crop and Pad")
    report_lines.append("   • Brightness/Contrast Adjustment")
    report_lines.append("")
    
    report_lines.append("4. BATCH SIZE & TRAINING:")
    report_lines.append("   For GTX 1650 (4GB VRAM):")
    report_lines.append("   • Recommended batch size: 8-16")
    report_lines.append("   • Use mixed precision training (FP16)")
    report_lines.append("   • Enable gradient checkpointing")
    report_lines.append("   • Use learning rate: 1e-4 to 5e-4")
    report_lines.append("")
    
    if 'quality_report' in stats:
        qr = stats['quality_report']
        report_lines.append("5. DATA QUALITY:")
        report_lines.append(f"   Current quality score: {qr['quality_score']:.2%}")
        if qr['quality_score'] >= 0.95:
            report_lines.append("   ✓ Excellent data quality - proceed with training")
        elif qr['quality_score'] >= 0.85:
            report_lines.append("   ✓ Good data quality - minor cleanup recommended")
        elif qr['quality_score'] >= 0.75:
            report_lines.append("   ⚠ Acceptable quality - review data issues before training")
        else:
            report_lines.append("   ✗ Poor data quality - significant cleanup required")
        report_lines.append("")
    
    report_lines.append("6. MODEL SELECTION:")
    report_lines.append("   Based on your configuration (GTX 1650 - 4GB VRAM):")
    report_lines.append("   • RECOMMENDED: ViT-Small + Lightweight Decoder")
    report_lines.append("   • Expected mIoU: 82-87%")
    report_lines.append("   • Inference time: 300-500ms per image")
    report_lines.append("   • Training time: 12-24 hours")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT".center(80))
    report_lines.append("=" * 80)
    
    # Write to file with UTF-8 encoding to support Unicode characters
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Text report created: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run EDA on floor plan dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset root directory (containing 'images' and 'annotations' folders)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations and reports (default: auto-generated)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=['cubicasa5k', 'roboflow'],
        default='cubicasa5k',
        help="Type of dataset (default: cubicasa5k)"
    )
    
    args = parser.parse_args()
    
    result = run_eda_analysis(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type
    )
    
    if result and result['success']:
        print("\n✓ EDA analysis completed successfully!")
        print(f"Output directory: {result['output_dir']}")
    else:
        print("\n✗ EDA analysis failed!")
        sys.exit(1)
