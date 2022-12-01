# ML of pXRDs using synthetic crystals

This repository contains the code used in our publication "Neural networks
trained on randomly generated crystals can classify space-groups of ICSD X-ray
diffractograms".

The repository contains the following components:

1. Optimized simulation and generation of crystals

    The code of the optimized simulation of powder XRDs (using numba LLVM
    just-in-time compilation) can be found in `./utils/simulation/`. This code
    is based on the implementation found in the `pymatgen` library.

    The code of the generation of synthetic crystals can be found in
    `./utils/generation/`.

    Before using our provided training script, these utilities should be
    installed as a package using `pip install -e ./utils/`. This will
    automatically install all required dependencies. The provided utilities are
    potentially also interesting to use for other projects concerning powder
    XRDs.

2. Distributed training

    The code for the distributed training architecture uses `tensorflow` with
    the distributed computing framework `ray`. The relevant script files can be
    found in `./training/`.

    To run the training script, the following additional dependencies must be
    installed:

    - `ray` > XXX
    - `tensorflow` > XXX

    Also make sure that the `CUDA` and `cuDNN` dependencies of `tensorflow` (the
    correct version that is compatible with your tensorflow version) are
    installed. We refer refer to the table available on
    https://www.tensorflow.org/install/source#tested_build_configurations.

# Documentation

## Optimized simulation of powder XRDs

## Generation of synthetic crystals

## Training