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

## Extracting statistics from the ICSD
In order to be able to generate synthetic crystals, some general statistics
about the occupation of the wyckoff positions for each space group need to be extracted
from the ICSD. If you only want to generate synthetic crystals without running your own
training experiments, you can use the statistical data contained in "training/prepared_training".
The required data can be loaded by using the function `load_dataset_info`...

- TODO: Can we decouple this? So that only the package is enough to already generate some 
synthetic crystals. Move `manage_dataset.py` to the package. Add a option to only read the statistics
without the training data. Add the statistics file of the prepared_dataset to the repository,
so that people can use it.

However, if you want to run your own ML experiments, you need to generate your own dataset
that also contains the required simulated diffractograms and crystals. 
They are needed to test the accuracy of the ML models.

For this, you first need to simulate patterns based on the ICSD database. For this, XXX
Then, you can run YYY

## Optimized simulation of powder XRDs
This repository provides various functions of simulating powder XRDs:

- Use function `from ml4pxrd_tools.simulation.simulation_core import get_pattern_optimized`
for fast simulation of the angles and intensities of all peaks in a given range. This 
uses a optimized version of the pymatgen implementation.
- Use function `from ml4pxrd_tools.simulation.simulation_smeared import get_smeared_patterns`
to simulate one or more smeared patterns (peaks convoluted with a Gaussian preak profile)
for a given structure object.
- Use function `from ml4pxrd_tools.simulation.simulation_smeared import get_synthetic_smeared_patterns`
to generate synthetic crystals and simulate pXRDs based on them. The synthetic crystal generation
is based on statistics extracted from the ICSD. 
    - TODO: What parameters to pass in from the get_dataset_info?

The functions `get_smeared_patterns` and `get_synthetic_smeared_patterns` calculate the FWHM of the gaussian
peak profiles using a random crystallite size uniformly sampled in the range `pymatgen_crystallite_size_gauss_min=20`
to `pymatgen_crystallite_size_gauss_max=100`. You can change the default range at the top of `simulation_smeared.py`.

## Generation of synthetic crystals
- TODO: How to use the output of the load dataset function to generate some crystals
- TODO: Mention that the preparation script of the next section needs to be run first for a lot of the functions
- TODO: Make it possible to use the generator functions with statistical data provided by us
- TODO: Mention what pickle interface is needed to load this one
- TODO: -> leave out the crystals etc. when loading the data; remove the unnecessary stuff.

## Training

In order to perform the training, you first need to generate a dataset. This is
based on the structures from the ICSD dataset. Thus, a license to the ICSD
database is needed.

- TODO: How to simulate the dataset
- TODO: Describe the benchmark and how to run it

In the beginning of the training script (`train_random_classifier.py`), you can
find options of the training including detailed explanations. While you should look
through all options, the following options need to be changed regardless:
- "path_to_patterns"
- "path_to_icsd_directory_local" or "path_to_icsd_directory_cluster"

- TODO: How to change configuration of computing nodes in the training script
- TODO: Command line options of the training script; alternatively, run it using the provided slurm scripts

- TODO: List the datasets that are used for validation
Used validation sets (TODO: Also, how are they named in the code? How are they named in TensorBoard?):
    - All ICSD entries
    - ICSD entries that match simulation parameters
    - Pre-computed random dataset (the one from the comparison script)
    - Gap between training and val acc that matches simulation parameters

The easiest way to track the progress and results of the training runs is to use `TensorBoard`.
Simly XXX

- TODO: Go through all TODOs in the whole project and fix them.
- TODO: LICENSE?

# Citing
To cite this repository, please refer to our publication:
XXX

# References
[1] Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A., & Ceder, G. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis (Version 2022.1.24) [Computer software]. https://doi.org/10.1016/j.commatsci.2012.10.028