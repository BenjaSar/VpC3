"""EDA visualization utilities."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_class_distribution(class_counts, title="Class Distribution"):
    """Plot class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    ax.bar(classes, counts)
    ax.set_xlabel('Class')
    ax.set_ylabel('Pixel Count')
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_image_size_distribution(sizes):
    """Plot distribution of image sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    size_counts = {}
    for size in sizes:
        size_str = f"{size[0]}x{size[1]}"
        size_counts[size_str] = size_counts.get(size_str, 0) + 1
    
    ax.bar(size_counts.keys(), size_counts.values())
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Count')
    ax.set_title('Image Size Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
