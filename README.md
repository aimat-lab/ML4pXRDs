# ML of pXRDs using synthetic crystals
This repository contains the code of the publication "Neural networks trained on
randomly generated crystals can classify space-groups of ICSD X-ray
diffractograms". It can be used to train machine learning models (e.g. for the
classification of space groups) on powder XRD patterns simulated on-the-fly from
synthetically generated random crystal structures.

If you have any problems using the provided software, if documentation is
missing, or if you find any bugs, feel free to contact us or add a new issue in
the Github repository.

The repository contains the following components:

1. Optimized simulation

    The code of the optimized simulation of powder XRDs (using numba LLVM
    just-in-time compilation) can be found in `./tools/simulation/`. This code
    is based on the implementation found in the
    [`pymatgen`](https://github.com/materialsproject/pymatgen) [1] library.

2. Generation of synthetic crystals

    The code of the generation of synthetic crystals can be found in
    `./tools/generation/`.

3. Distributed training

    The code of the distributed training architecture uses `tensorflow` with
    the distributed computing framework `ray`. The relevant script files can be
    found in `./training/`.

# Documentation
## Getting started

For convenience and because the provided utilities are potentially also
interesting to use for other projects concerning powder XRDs, the code for the
simulation of pXRDs and generation of synthetic crystals is provided as a
package. Before training, this should be installed, ideally in a separate
virtual environment or anaconda environment. Therefore, call pip in the root of
the repository:

```
pip install -e .
```

This will further install all required dependencies. 

To run the training script, the following additional dependencies must be
installed:

- `ray` >= 1.9.1
- `tensorflow` >= 2.0.0

Also make sure that the `CUDA` and `cuDNN` dependencies of `tensorflow` (the
correct version that are compatible with your tensorflow version) are installed.
We refer to the table available on
https://www.tensorflow.org/install/source#tested_build_configurations.

## Loading statistics of the ICSD
In order to be able to generate synthetic crystals, some general statistics
about the occupation of the wyckoff positions for each space group need to be
extracted from the ICSD. If you only want to generate synthetic crystals (and
simulate pXRDs based on them) without running your own training experiments, you
can use the statistical data contained in "./public_statistics". The required
data can be loaded by using the function `load_dataset_info` with parameter
`load_public_statistics_only=True`. The returned objects can then be passed to
the respective functions to generate crystals and simulate pXRDs (see below).

- TODO: Mention what pickle interface is needed to load this one
- TODO: Full code to load statistics

## Generating synthetic crystals
- TODO: Make one example to call the function

## Simulating pXRDs
This repository provides various functions of simulating powder XRDs:

- Use function `from ml4pxrd_tools.simulation.simulation_core import
get_pattern_optimized` for fast simulation of the angles and intensities of all
peaks in a given range. This uses a optimized version of the pymatgen
implementation.
- Use function `from ml4pxrd_tools.simulation.simulation_smeared import get_smeared_patterns`
to simulate one or more smeared patterns (peaks convoluted with a Gaussian preak profile)
for a given structure object.
- Use function `from ml4pxrd_tools.simulation.simulation_smeared import get_synthetic_smeared_patterns`
to generate synthetic crystals and simulate pXRDs based on them. The synthetic crystal generation
is based on statistics extracted from the ICSD. 
    
- TODO: Similar to above for the generation of crystals, make one example function call

The functions `get_smeared_patterns` and `get_synthetic_smeared_patterns`
calculate the FWHM of the gaussian peak profiles using a random crystallite size
uniformly sampled in the range `pymatgen_crystallite_size_gauss_min=20` to
`pymatgen_crystallite_size_gauss_max=100`. You can change the default range at
the top of `simulation_smeared.py`.

## Training
### Preparing the dataset
If you want to run your own ML experiments, you need to generate your own
dataset from the ICSD that also contains the required simulated diffractograms
and crystals. They are needed to test the accuracy of the ML models.

In order to generate a dataset, a license for the ICSD database is needed.
If you have the license and downloaded the database, you need to XXX.

- TODO: How to simulate the ICSD
- TODO: How to generate statistics and dataset split (will take a while)

In the beginning of the training script (`train_random_classifier.py`), you can
find options of the training including detailed explanations. While you should
look through all options, the following options need to be changed regardless:

- "path_to_patterns"
- "path_to_icsd_directory_local" or "path_to_icsd_directory_cluster"

- TODO: Change environment name in slurm scripts

- TODO: How to simulate the dataset

- TODO: How to change configuration of computing nodes in the training script
- TODO: Command line options of the training script; alternatively, run it using
  the provided slurm scripts

- TODO: List the datasets that are used for validation
Used validation sets (TODO: Also, how are they named in the code? How are they named in TensorBoard?):
    - All ICSD entries
    - ICSD entries that match simulation parameters
    - Pre-computed random dataset (the one from the comparison script)
    - Gap between training and val acc that matches simulation parameters

The easiest way to track the progress and results of the training runs is to use
`TensorBoard`. Simly XXX

- TODO: Go through all TODOs in the whole project and fix them.
- TODO: LICENSE?

# Citing
To cite this repository, please refer to our publication:
XXX

# References
[1] Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A., & Ceder, G. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis (Version 2022.1.24) [Computer software]. https://doi.org/10.1016/j.commatsci.2012.10.028