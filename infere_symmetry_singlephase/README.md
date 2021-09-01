# Inference of Bravais Lattice Type from XRD Patterns

This projects aims to extract the bravais lattice type from a given xrd pattern.
This is done with different types of neural networks (fully connected, convolutional and lstm at the moment)
with varying accuracy. The goal of this project is mostly testing purposes, the aim 
is not to be able to infere the bravais lattice type from actual noisy experimental data.

## The dataset
The XRD patterns in the dataset for training were simulated using the python library `pymatgen`.
The structures were taken from the `Crystallography Open Database` (476894 structures).
You can use the script `create_dataset.py` to simulate your own dataset (you need to download the CIF files first). The simulation, however,
takes a couple hundred CPU-hours. If you just want to play around with the dataset, I can upload it somewhere, just contact me.