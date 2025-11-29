"""
Setup script for CubiCasa5K Room Detection
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README_COMPLETO.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="floorplan-classifier",
    version="1.0.0",
    author="Vision por Computadora 3 - Maestría",
    author_email="jorge.cuenca@unillanos.edu.co",
    description="Sistema avanzado de detección de habitaciones y estimación de áreas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenjaSar/floorplan-classifier/",
    project_urls={
        "Index": "https://github.com/BenjaSar/floorplan-classifier/blob/main/README.md",
        "Documentation": "https://github.com/BenjaSar/floorplan-classifier/docs",
        "Source Code": "https://github.com/BenjaSar/floorplan-classifier/tree/swin_maskrcnn",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.13",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cubicasa-train=scripts.train:main",
            "cubicasa-eval=scripts.evaluate:main",
            "cubicasa-infer=scripts.inference:main",
            "cubicasa-eda=scripts.eda_analysis:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "computer vision",
        "deep learning",
        "room detection",
        "floor plan analysis",
        "swin transformer",
        "mask rcnn",
        "instance segmentation",
        "cubicasa5k",
        "mlops",
    ],
)
