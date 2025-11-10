from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vit_floorplan",
    version="0.1.0",
    author="FS, Alejandro Lloveras, Jorge Cuenca",
    author_email="fs@example.com",
    github_username="BenjaSar",
    description="Vision Transformer for floorplan segmentation and classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)
