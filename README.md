# ML of pXRDs using synthetic crystals

This repository contains the code of the publication "Neural networks trained on
randomly generated crystals can classify space-groups of ICSD X-ray
diffractograms". It can be used to train machine learning models (e.g. for the
classification of space groups) on powder XRD patterns simulated on-the-fly from
synthetically generated random crystal structures.

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

Before using our provided training script, the code for the simulation of pXRDs
and generation of synthetic crystals should be installed as a package using `pip
install -e ./tools/`, ideally in a separate python of anaconda environment. This
will automatically install all required dependencies. The provided utilities are
potentially also interesting to use for other projects concerning powder XRDs.

To run the training script, the following additional dependencies must be
installed:

- `ray` >= 1.9.1
- `tensorflow` >= 2.0.0

Also make sure that the `CUDA` and `cuDNN` dependencies of `tensorflow` (the
correct version that is compatible with your tensorflow version) are
installed. We refer refer to the table available on
https://www.tensorflow.org/install/source#tested_build_configurations.

## Optimized simulation of powder XRDs

## Generation of synthetic crystals

## Training

# References
[1] Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A., & Ceder, G. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis (Version 2022.1.24) [Computer software]. https://doi.org/10.1016/j.commatsci.2012.10.028