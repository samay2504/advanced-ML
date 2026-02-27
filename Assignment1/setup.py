"""Setup script for gbt package."""
from setuptools import setup, find_packages

setup(
    name="gbt",
    version="0.1.0",
    description="Gradient Boosting Trees from scratch (Algorithms 10.3 & 10.4)",
    author="AML Project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
        ]
    },
)
