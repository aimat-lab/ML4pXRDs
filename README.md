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
virtual environment or anaconda environment. We tested the package for python
3.8.0 on Ubuntu, but it should also work for other python versions and operating
systems.

Call pip in the root of the repository:

```
pip install -e .
```

This will further install all required dependencies. 

To further run the training script and some of the analysis scripts in
`./training/analysis`, the following additional dependencies can be installed
using pip:

- `ray`
- `psutil`
- `ase`
- `tensorflow`

We tested and recommend tensorflow version 2.10.0. Also, make sure that the
`CUDA` and `cuDNN` dependencies of `tensorflow` (the correct version that are
compatible with your tensorflow version) are installed. We refer to the table
available on
https://www.tensorflow.org/install/source#tested_build_configurations. For
tensorflow 2.10.0, you can simply install the required `CUDA` and `cuDNN`
dependencies using conda:

```
conda install -c conda-forge cudatoolkit==11.2.0
conda install -c conda-forge cudnn==8.1.0.77
```

## Loading statistics of the ICSD
In order to be able to generate synthetic crystals, some general statistics
about the occupation of the wyckoff positions for each space group need to be
extracted from the ICSD. If you only want to generate synthetic crystals (and
simulate pXRDs based on them) without running your own training experiments, you
can use the statistical data provided by us in "./public_statistics". 

The required data can be loaded by using the function `load_dataset_info` with
parameter `load_public_statistics_only=True`. The returned objects can then be
passed to the respective functions to generate crystals and simulate pXRDs (see
below). We refer to section `Training` if you want to create your own dataset 
and extract your own statistics from the ICSD.

```python
(
    probability_per_spg_per_element,
    probability_per_spg_per_element_per_wyckoff,
    NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element,
    denseness_factors_density_per_spg,
    denseness_factors_conditional_sampler_seeds_per_spg,
    lattice_paras_density_per_lattice_type,
    per_element,
    represented_spgs,
    probability_per_spg,
) = load_dataset_info(load_public_statistics_only=True)
```

## Generating synthetic crystals

After loading the statistics, you can use the statistics to generate synthetic structures
of a given space group:

```python
structures = generate_structures(
    125,
    N=1,
    probability_per_spg_per_element=probability_per_spg_per_element,
    probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
    NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
    denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
    lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
)
```

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

Here is an example of how to call `get_synthetic_smeared_patterns` using the
statistics loaded using `load_dataset_info` (here for space group 125):

```python
patterns, labels = get_synthetic_smeared_patterns(
    [125],
    N_structures_per_spg=5,
    wavelength=1.5406,
    two_theta_range=(5, 90),
    N=8501,
    NO_corn_sizes=1,
    probability_per_spg_per_element=probability_per_spg_per_element,
    probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
    NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
    denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
    lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
)
```    

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

- TODO: icsd_simulator.py:
This script can be used to simulate pXRD patterns of the ICSD database. Before
running this script, make sure that you change the variables on top of this
script file, the file `simulation_worker.py`, and `simulation_smeared.py`.

- TODO: submit_icsd_simulation_slurm.slr
- TODO: Change in the submit script how the conda env is loaded!
- Write here that this needs to be changed, first (in the script file)

To generate a new dataset with prototype-based split, you first have to change
`path_to_icsd_directory_cluster` or `path_to_icsd_directory_local` (depends on
if you run this script on a cluster using slurm or not) in this script. It
should point to your directory containing the ICSD database. Furthermore, you
first need to run the simulation of the ICSD data (see README.md) and point
`path_to_patterns` (see below) to the directory containing your simulated 
patterns.
Then, you can run this file to generate the dataset: `python manage_dataset.py`

In the beginning of the training script (`train_random_classifier.py`), you can
find options of the training including detailed explanations. While you should
look through all options, the following options need to be changed regardless:

- "path_to_patterns"
- "path_to_icsd_directory_local" or "path_to_icsd_directory_cluster"

- TODO: Talk about submit scripts in general
- TODO: Change environment name in slurm scripts
- Change method in script of how environment is activated
- Fixed paths?
- submit_head_only.slr

- TODO: How to change configuration of computing nodes in the training script
- TODO: Command line options of the training script; alternatively, run it using
  the provided slurm scripts

- TODO: Talk about created run directory

- TODO: List the datasets that are used for validation
Used validation sets (TODO: Also, how are they named in the code? How are they named in TensorBoard?):
    - All ICSD entries
    - ICSD entries that match simulation parameters
    - Pre-computed random dataset (the one from the comparison script)
    - Gap between training and val acc that matches simulation parameters

The easiest way to track the progress and results of the training runs is to use
`TensorBoard`. Simply navigate to the run directory in your terminal and execute
`tensorboard --logdir .`.

# Citing
To cite this repository, please refer to our publication:
- TODO: Add reference to arxiv paper

# References
[1] Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A., & Ceder, G. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis (Version 2022.1.24) [Computer software]. https://doi.org/10.1016/j.commatsci.2012.10.028