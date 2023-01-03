from setuptools import setup
from setuptools import find_packages

setup(
    name="ML4pXRDs_tools",
    version="1.0.0",
    description="Includes code for the optimized simulation of pXRDs and the generation of synthetic crystals.",
    url="https://github.com/aimat-lab/ML4pXRDs",
    author="Henrik Schopmans",
    author_email="henrik.schopmans@kit.edu",
    packages=find_packages(exclude=["training"]),
    install_requires=[
        "numpy",
        "pyxtal",
        "pymatgen",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "numba",
        "statsmodels",
    ],
)

# Additionally needed for training scripts:

# pip: ray
# pip: tensorflow_addons
# pip: psutil
# pip: ase

# Our recommendation:
# conda: cudatoolkit conda-forge 11.2.0
# conda: cudnn conda-forge 8.1.0.77
# pip: tensorflow 2.10.0
