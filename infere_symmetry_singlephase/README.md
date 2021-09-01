# Inference of Crystal Symmetry from XRD Diffractograms

This part of the project aims to extract the bravais lattice type (14 categories) from a given XRD pattern.
This is done with different types of neural networks (fully connected, convolutional and lstm at the moment)
with varying accuracy. The goal, for now, is a proof of concept. 

Actual noisy experimental data with background and space group prediction will come later.

## The dataset
The XRD patterns in the dataset for training were simulated using the python library `pymatgen`.
The structures were taken from the ICSD database (239679 structures after clearning by Timo Sommer (thanks!)).

The script `create_dataset.py` can be used to simulate the dataset from the cif files of the database.