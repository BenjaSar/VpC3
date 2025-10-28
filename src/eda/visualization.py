"""
Visualization Engine for EDA
Creates comprehensive visualizations for floor plan dataset analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    Create visualizations for EDA
    """
    
    def __init__(self, output_dir: str = "./eda_output"):
        """
        Initialize visualization engine
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 10
        logger.info(f"Visualization output directory: {self.output_dir}")
    
    def plot_image_dimensions(self, df_stats: pd.DataFrame):
        """
        Plot image dimensions distribution
        
        Args:
            df_stats: DataFrame with image statistics
        """
        logger.info("Creating image dimensions plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Image Dimensions Distribution', fontsize=16, fontweight='bold')
        
        # Width distribution
        axes[0, 0].hist(df_stats['width'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Width Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].axvline(df_stats['width'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df_stats['width'].mean():.0f}")
        axes[0, 0].axvline(df_stats['width'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df_stats['width'].median():.0f}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(df_stats['height'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Height (pixels)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Height Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].axvline(df_stats['height'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df_stats['height'].mean():.0f}")
        axes[0, 1].axvline(df_stats['height'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df_stats['height'].median():.0f}")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Aspect ratio distribution
        axes[1, 0].hist(df_stats['aspect_ratio'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].axvline(df_stats['aspect_ratio'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df_stats['aspect_ratio'].mean():.2f}")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: Width vs Height
        scatter = axes[1, 1].scatter(df_stats['width'], df_stats['height'], 
                                     c=df_stats['file_size_mb'], cmap='viridis', 
                                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[1, 1].set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Width vs Height (colored by file size)', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('File Size (MB)', fontsize=10, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "01_image_dimensions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved image dimensions plot to {output_path}")
        plt.close()
    
    def plot_file_size_distribution(self, df_stats: pd.DataFrame):
        """
        Plot file size distribution
        
        Args:
            df_stats: DataFrame with image statistics
        """
        logger.info("Creating file size distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('File Size Distribution', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0].hist(df_stats['file_size_mb'], bins=30, color='purple', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('File Size (MB)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('File Size Histogram', fontsize=12, fontweight='bold')
        axes[0].axvline(df_stats['file_size_mb'].mean(), color='red', linestyle='--', linewidth=2, 
                        label=f"Mean: {df_stats['file_size_mb'].mean():.2f} MB")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df_stats['file_size_mb'], vert=True)
        axes[1].set_ylabel('File Size (MB)', fontsize=11, fontweight='bold')
        axes[1].set_title('File Size Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "02_file_size_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved file size distribution plot to {output_path}")
        plt.close()
    
    def plot_class_distribution(self, annotation_analysis: Dict, imbalance_analysis: Dict):
        """
        Plot class distribution and imbalance
        
        Args:
            annotation_analysis: Results from analyze_annotations()
            imbalance_analysis: Results from analyze_class_imbalance()
        """
        logger.info("Creating class distribution plot...")
        
        class_stats = imbalance_analysis.get('class_statistics', {})
        
        # Check if we have any class data
        if not class_stats or len(class_stats) == 0:
            logger.warning("No class distribution data available - creating placeholder plot")
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.text(0.5, 0.5, 'No Class Distribution Data\n\nAnnotations appear to be empty or invalid', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.axis('off')
            fig.suptitle('Class Distribution and Imbalance Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            output_path = self.output_dir / "03_class_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved placeholder class distribution plot to {output_path}")
            plt.close()
            return
        
        # Extract data with error handling
        try:
            classes = sorted([int(c) for c in class_stats.keys()])
            percentages = [class_stats[str(c)]['percentage'] for c in classes]
            weights = [class_stats[str(c)]['weight'] for c in classes]
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error extracting class statistics: {e}")
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.text(0.5, 0.5, 'Error Processing Class Distribution\n\nData structure issue detected', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax.axis('off')
            fig.suptitle('Class Distribution and Imbalance Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            output_path = self.output_dir / "03_class_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error placeholder plot to {output_path}")
            plt.close()
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Class Distribution and Imbalance Analysis', fontsize=16, fontweight='bold')
        
        # Class percentage distribution
        bars1 = axes[0, 0].bar(classes, percentages, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Class ID', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Class Distribution (% of pixels)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Class weights (for weighted loss)
        bars2 = axes[0, 1].bar(classes, weights, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Class ID', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Weight', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Class Weights (for balanced training)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Imbalance ratio
        imbalance_ratios = [max(percentages) / max(p, 0.001) for p in percentages]
        bars3 = axes[1, 0].bar(classes, imbalance_ratios, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Class ID', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Imbalance Ratio', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Imbalance Ratio per Class', fontsize=12, fontweight='bold')
        axes[1, 0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Balanced')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Top 10 classes
        top_classes = sorted(class_stats.items(), 
                            key=lambda x: x[1]['percentage'], 
                            reverse=True)[:10]
        top_class_ids = [int(c[0]) for c in top_classes]
        top_percentages = [c[1]['percentage'] for c in top_classes]
        
        bars4 = axes[1, 1].barh(range(len(top_class_ids)), top_percentages, 
                               color='mediumpurple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_class_ids)))
        axes[1, 1].set_yticklabels([f'Class {c}' for c in top_class_ids])
        axes[1, 1].set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Top 10 Most Common Classes', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars4, top_percentages)):
            axes[1, 1].text(val, i, f' {val:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "03_class_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution plot to {output_path}")
        plt.close()
    
    def plot_pixel_statistics(self, annotation_analysis: Dict):
        """
        Plot pixel-level statistics
        
        Args:
            annotation_analysis: Results from analyze_annotations()
        """
        logger.info("Creating pixel statistics plot...")
        
        pixel_stats = annotation_analysis['pixel_statistics']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Pixel-Level Statistics', fontsize=16, fontweight='bold')
        
        # Pie chart: Pixel distribution
        labels = ['Background', 'Annotated']
        sizes = [pixel_stats['background_pixels'], pixel_stats['annotated_pixels']]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0.05)
        
        # Check if we have valid data for pie chart
        if sum(sizes) > 0 and all(s >= 0 for s in sizes):
            axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                       shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            axes[0].set_title('Background vs Annotated Pixels', fontsize=12, fontweight='bold')
        else:
            # If no valid data, display message
            axes[0].text(0.5, 0.5, 'No valid pixel data\nfor pie chart', 
                        ha='center', va='center', fontsize=14, transform=axes[0].transAxes)
            axes[0].set_title('Background vs Annotated Pixels', fontsize=12, fontweight='bold')
            axes[0].axis('off')
        
        # Bar chart: Pixel statistics
        stats_labels = ['Total\nPixels', 'Valid\nPixels', 'Background\nPixels', 'Annotated\nPixels']
        stats_values = [
            pixel_stats['total_pixels'],
            pixel_stats['valid_pixels'],
            pixel_stats['background_pixels'],
            pixel_stats['annotated_pixels']
        ]
        
        # Convert to millions for readability
        stats_values_millions = [v / 1e6 for v in stats_values]
        
        bars = axes[1].bar(stats_labels, stats_values_millions, 
                          color=['lightcoral', 'steelblue', 'lightgray', 'lightgreen'],
                          edgecolor='black', alpha=0.7)
        axes[1].set_ylabel('Pixels (Millions)', fontsize=11, fontweight='bold')
        axes[1].set_title('Pixel Statistics', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, stats_values_millions):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "04_pixel_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved pixel statistics plot to {output_path}")
        plt.close()
    
    def plot_mask_dimensions(self, annotation_analysis: Dict):
        """
        Plot mask dimensions distribution
        
        Args:
            annotation_analysis: Results from analyze_annotations()
        """
        logger.info("Creating mask dimensions plot...")
        
        mask_dims = annotation_analysis['mask_dimensions']
        widths = mask_dims['widths']
        heights = mask_dims['heights']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Mask Dimensions Distribution', fontsize=16, fontweight='bold')
        
        # Width distribution
        axes[0, 0].hist(widths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Mask Width Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, 
                          label=f"Mean: {np.mean(widths):.0f}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(heights, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Height (pixels)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Mask Height Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, 
                          label=f"Mean: {np.mean(heights):.0f}")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter: Width vs Height
        axes[1, 0].scatter(widths, heights, alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
        axes[1, 0].set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Mask Width vs Height', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Resolution statistics
        resolutions = [(w, h) for w, h in zip(widths, heights)]
        resolution_counts = pd.Series(resolutions).value_counts()
        
        axes[1, 1].barh(range(len(resolution_counts.index[:10])), 
                       resolution_counts.values[:10],
                       color='mediumpurple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_yticks(range(len(resolution_counts.index[:10])))
        axes[1, 1].set_yticklabels([f"{w}x{h}" for w, h in resolution_counts.index[:10]], fontsize=9)
        axes[1, 1].set_xlabel('Count', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Top 10 Most Common Resolutions', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_path = self.output_dir / "05_mask_dimensions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved mask dimensions plot to {output_path}")
        plt.close()
    
    def plot_quality_report(self, quality_report: Dict):
        """
        Plot data quality metrics
        
        Args:
            quality_report: Quality report from generate_quality_report()
        """
        logger.info("Creating quality report visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # Quality metrics gauge
        metrics = ['Data\nCompleteness', 'Image-Annotation\nMatch', 'Overall\nQuality']
        values = [
            quality_report['data_completeness'] * 100,
            quality_report['image_annotation_match'] * 100,
            quality_report['quality_score'] * 100
        ]
        colors_gauge = ['green' if v >= 85 else 'orange' if v >= 75 else 'red' for v in values]
        
        bars = axes[0, 0].bar(metrics, values, color=colors_gauge, edgecolor='black', alpha=0.7)
        axes[0, 0].set_ylabel('Score (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Quality Metrics', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim([0, 100])
        axes[0, 0].axhline(85, color='green', linestyle='--', alpha=0.5, label='Excellent (85%)')
        axes[0, 0].axhline(75, color='orange', linestyle='--', alpha=0.5, label='Acceptable (75%)')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Status indicator
        status = quality_report['summary']['status']
        status_color = {
            'EXCELLENT': 'green',
            'GOOD': 'lightgreen',
            'ACCEPTABLE': 'yellow',
            'POOR': 'red'
        }
        
        axes[0, 1].text(0.5, 0.7, f"Status: {status}", 
                       ha='center', va='center', fontsize=20, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=status_color.get(status, 'gray'), alpha=0.7))
        axes[0, 1].text(0.5, 0.3, f"Quality Score: {quality_report['quality_score']:.1%}",
                       ha='center', va='center', fontsize=14)
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Overall Status', fontsize=12, fontweight='bold')
        
        # Issues and recommendations
        issues_text = "Issues Found:\n" + "\n".join([f"• {issue}" for issue in quality_report['issues']]) \
                     if quality_report['issues'] else "No issues found!"
        
        axes[1, 0].text(0.05, 0.95, issues_text, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Issues & Warnings', fontsize=12, fontweight='bold')
        
        # Recommendations
        recommendations_text = "Recommendations:\n" + "\n".join([f"• {rec}" for rec in quality_report['recommendations']]) \
                              if quality_report['recommendations'] else "No recommendations!"
        
        axes[1, 1].text(0.05, 0.95, recommendations_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Recommendations', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "06_quality_report.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved quality report plot to {output_path}")
        plt.close()
    
    def plot_sample_floor_plans(self, images_path: str, annotations_path: str, num_samples: int = 6):
        """
        Plot sample floor plans with their annotations
        
        Args:
            images_path: Path to images directory
            annotations_path: Path to annotations directory
            num_samples: Number of samples to display
        """
        logger.info(f"Creating sample floor plans visualization (n={num_samples})...")
        
        import cv2
        from pathlib import Path
        
        images_dir = Path(images_path)
        annotations_dir = Path(annotations_path)
        
        image_files = sorted(list(images_dir.glob('*.*')))[:num_samples]
        
        # Create grid
        n_cols = 2
        n_rows = (len(image_files) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(16, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Sample Floor Plans (n={len(image_files)})', fontsize=16, fontweight='bold')
        
        for idx, img_path in enumerate(image_files):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load corresponding annotation
            mask_path = annotations_dir / (img_path.stem + '.png')
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            row = idx // n_cols
            col = (idx % n_cols) * 2
            
            # Display image
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(f'Image: {img_path.name}', fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
            
            # Display mask with colormap
            if mask is not None:
                mask_colored = plt.cm.tab20(mask.astype(float) / mask.max()) if mask.max() > 0 else mask
                axes[row, col + 1].imshow(mask_colored)
                axes[row, col + 1].set_title(f'Annotation: {mask_path.name}', fontsize=10, fontweight='bold')
                axes[row, col + 1].axis('off')
        
        # Hide empty subplots
        for idx in range(len(image_files), n_rows * n_cols):
            row = idx // n_cols
            col = (idx % n_cols) * 2
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / "07_sample_floor_plans.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sample floor plans plot to {output_path}")
        plt.close()
    
    def create_eda_report(self, df_stats: pd.DataFrame, annotation_analysis: Dict, 
                         imbalance_analysis: Dict, quality_report: Dict,
                         images_path: str, annotations_path: str):
        """
        Create all EDA visualizations
        
        Args:
            df_stats: Image statistics DataFrame
            annotation_analysis: Annotation analysis results
            imbalance_analysis: Class imbalance analysis results
            quality_report: Quality report
            images_path: Path to images directory
            annotations_path: Path to annotations directory
        """
        logger.info("Creating comprehensive EDA report...")
        
        # Create all plots
        self.plot_image_dimensions(df_stats)
        self.plot_file_size_distribution(df_stats)
        self.plot_class_distribution(annotation_analysis, imbalance_analysis)
        self.plot_pixel_statistics(annotation_analysis)
        self.plot_mask_dimensions(annotation_analysis)
        self.plot_quality_report(quality_report)
        self.plot_sample_floor_plans(images_path, annotations_path, num_samples=6)
        
        logger.info("EDA report creation completed!")
        return self.output_dir
