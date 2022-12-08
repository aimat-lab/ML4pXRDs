# ML of pXRDs using synthetic crystals

This repository contains the code of the publication "Neural networks trained on
randomly generated crystals can classify space-groups of ICSD X-ray
diffractograms". It can be used to train machine learning models (e.g. for the
classification of space groups) on powder XRD patterns simulated on-the-fly from
synthetically generated random crystal structures.

If you have any problems using the provided software or if you find any bugs,
feel free to contact us or add a new issue in the Github repository.

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
We refer refer to the table available on
https://www.tensorflow.org/install/source#tested_build_configurations.

## Optimized simulation of powder XRDs
- TODO: Change crystallite size range (random sampling)

## Generation of synthetic crystals
- TODO: How to use the output of the load dataset function to generate some crystals
=> refer to the __main__ part of the script

## Training

In order to perform the training, you first need to generate a dataset. This is
based on the structures from the ICSD dataset. Thus, a license to the ICSD
database is needed.

- TODO: How to simulate the dataset

In the beginning of the training script (`train_random_classifier.py`), you can
find options of the training including detailed explanations. While you should look
through all options, the following options need to be changed regardless:
- "path_to_patterns"
- "path_to_icsd_directory_local" or "path_to_icsd_directory_cluster"

- TODO: Change configuration of computing nodes in the training script
- TODO: Command line options of the training script

- TODO: List the datasets that are used for validation
Used validation sets:
    - All ICSD entries
    - ICSD entries that match simulation parameters
    - Pre-computed random dataset (the one from the comparison script)
    - Gap between training and val acc that matches simulation parameters

- TODO: LICENSE?
- TODO: Go through all TODOs in the whole project and fix them.

# References
[1] Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A., & Ceder, G. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis (Version 2022.1.24) [Computer software]. https://doi.org/10.1016/j.commatsci.2012.10.028