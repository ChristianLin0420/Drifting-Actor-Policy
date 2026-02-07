"""
Drifting-VLA Package Setup
==========================

Installation:
    pip install -e .
    pip install -e ".[dev]"  # with development dependencies
"""

from setuptools import setup, find_packages

setup(
    name="drifting-vla",
    version="0.1.0",
    description="Drifting-VLA: Vision-Language-Action Policy with Drifting Model",
    author="Drifting-VLA Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "hydra-core>=1.3.2",
        "wandb>=0.16.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "h5py>=3.10.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=24.2.0",
            "ruff>=0.2.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drifting-train=scripts.train:main",
            "drifting-eval=scripts.evaluate:main",
        ],
    },
)


