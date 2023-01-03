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

# Additional for training scripts:
# cudatoolkit conda-forge 11.2.0
# cudnn conda-forge 8.1.0.77
# tensorflow>2.5.0???
# ray
# keras; automatically installed?
# tensorflow_addons; also possible from pip? Currently locally installed.
# psutil
# ase
