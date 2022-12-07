from setuptools import setup

setup(
    name="ML4pXRDs_tools",
    version="1.0.0",
    description="Includes code for the fast simulation of pXRDs and the generation of synthetic crystals.",
    url="https://github.com/aimat-lab/ML4pXRDs",
    author="Henrik Schopmans",
    author_email="henrik.schopmans@kit.edu",
    packages=["ml4pxrd_tools"],
    install_requires=[
        "numpy",
    ],
)

# TODO: Update the dependencies
