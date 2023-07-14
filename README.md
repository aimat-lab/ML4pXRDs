# ML for pXRDs using synthetic crystals
This repository contains the code of the publication ["Neural networks trained on
synthetically generated crystals can extract structural information from ICSD
powder X-ray diffractograms"](https://arxiv.org/abs/2303.11699). It can be used to train machine learning models
(e.g., for the classification of space groups) on powder XRD diffractograms
simulated on-the-fly from synthetically generated random crystal structures.

You can find details about this project in our [`paper`](https://arxiv.org/abs/2303.11699). If you want to cite our work, you can use the provided bibtex file [CITATION.bib](CITATION.bib).

If you have any problems using the provided software, if documentation is
missing, or if you find any bugs, feel free to add a new issue on GitHub.

The repository contains the following components:

1. Optimized simulation

    The code of the optimized simulation of powder XRDs (using numba LLVM
    just-in-time compilation) can be found in `./ml4pxrd_tools/simulation/`. This code
    is based on the implementation found in the
    [`pymatgen`](https://github.com/materialsproject/pymatgen) library.

2. Generation of synthetic crystals

    The code of the generation of synthetic crystals can be found in
    `./ml4pxrd_tools/generation/`.

3. Distributed training

    The code of the distributed training architecture uses `tensorflow` with
    the distributed computing framework `ray`. The relevant script files can be
    found in `./training/`.

# Documentation
## Getting started
For convenience, the code for the optimized simulation of pXRDs and generation
of synthetic crystals is provided as a package called `ml4pxrd_tools`. Before
training, this should be installed, ideally in a separate virtual environment or
anaconda environment. We tested the package with python 3.8.0 on Ubuntu, but it
should also work for other python versions and operating systems.

To install the package, call pip in the root of the repository:

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
- `tensorflow-addons`

We tested and recommend TensorFlow version 2.10.0. Also, make sure that the
`CUDA` and `cuDNN` dependencies of `tensorflow` are installed and that the
versions are compatible (we refer to the table available at
https://www.tensorflow.org/install/source#tested_build_configurations). For
TensorFlow 2.10.0, you can simply install the required `CUDA` and `cuDNN`
dependencies using conda:

```
conda install -c conda-forge cudatoolkit==11.2.0
conda install -c conda-forge cudnn==8.1.0.77
```

## Loading statistics of the ICSD
In order to be able to generate synthetic crystals, some general statistics
(e.g., about the occupation of the Wyckoff positions for each space group) need
to be extracted from the ICSD. If you only want to generate synthetic crystals
(and simulate pXRDs based on them) without running your own training
experiments, you can use the statistical data provided by us in
`./public_statistics`. We refer to section `Training` of this README if you want
to create your own dataset and extract your own statistics from the ICSD.

The required data can be loaded using the function
`ml4pxrd_tools.manage_dataset.load_dataset_info` with parameter
`load_public_statistics_only=True`. The returned objects can then be passed to
the respective functions to generate synthetic crystals and simulate pXRDs (see
below). 

```python
from ml4pxrd_tools.manage_dataset import load_dataset_info

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

After loading the statistics, you can use the statistics to generate synthetic
structures of a given space group (here for space group 125):

```python
from ml4pxrd_tools.generation.structure_generation import generate_structures

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
This repository provides various functions to simulate powder XRD diffractograms:

- Use function `ml4pxrd_tools.simulation.simulation_core.get_pattern_optimized`
for fast simulation of the angles and intensities of all peaks in a given
$2\theta$ range. This uses an optimized version of the pymatgen implementation.
- Use function `ml4pxrd_tools.simulation.simulation_smeared.get_smeared_patterns`
to simulate one or more smeared patterns (peaks convoluted with a Gaussian preak profile)
for a given structure object.
- Use function `ml4pxrd_tools.simulation.simulation_smeared.get_synthetic_smeared_patterns`
to generate synthetic crystals and simulate pXRDs based on them.

Here is an example of how to call `get_synthetic_smeared_patterns` using the
statistics loaded using `load_dataset_info` (see above):

```python
from ml4pxrd_tools.simulation.simulation_smeared import get_synthetic_smeared_patterns

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
calculate the FWHM of the gaussian peak profiles using the Scherrer equation
with a random crystallite size uniformly sampled in the range
`pymatgen_crystallite_size_gauss_min=20` to
`pymatgen_crystallite_size_gauss_max=100` (in nm). You can change the default
range at the top of script file
`./ml4pxrd_tools/simulation/simulation_smeared.py`.

## Training
You can find the weights of our largest model (ResNet-101) trained using
synthetic crystals and the weights of the ResNet-50 trained with 
experimental imperfections in our [latest release](https://github.com/aimat-lab/ML4pXRDs/releases/tag/v1.0).

### Pre-simulate patterns for testing
If you want to run your own ML experiments, you need to generate your own
dataset from the ICSD that contains the required simulated diffractograms
and crystals. This is needed to test the accuracy of the ML models.

In order to generate a dataset, a license for the ICSD database is needed. If
you have the license and downloaded the database, you need to first simulate
powder diffractograms based on the ICSD crystals. This can be accomplished by running
the script `./ml4pxrd_tools/simulation/icsd_simulator.py`. Before running this
script, make sure that you change the variables at the top of this script file,
of the file `simulation_worker.py`, and of `simulation_smeared.py`.

Instead of running the script directly, you can also use the provided slurm
script `submit_icsd_simulation_slurm.slr` to run it on a cluster. Make sure to
adapt it to your cluster first and potentially change the path to your `.bashrc`
file and the name of your anaconda environment.

As a point of reference, it takes ~14 hours to simulate the full ICSD on 8 cores.

### Extract statistics and generate dataset split
To generate a new dataset with prototype-based split using the just simulated
patterns, you can use the script `./ml4pxrd_tools/manage_dataset.py`. Please
first change the variables at the top of this script file. Then, you can
generate the dataset and extract the statistics: 

```bash
python manage_dataset.py
```

This will take a while (~5 hours). Finally, you can find the prepared dataset
including the statistics in the directory `./prepared_dataset`.

### Run experiments
At the top of the training script (`./trainig/train_random_classifier.py`), you
can find some variables / options of the training experiment including detailed
explanations. While you should look through all options, the following options
always need to be changed:

- `path_to_patterns`
- `path_to_icsd_directory_local` or `path_to_icsd_directory_cluster`

Furthermore, you might want to change the used model (see line `model =
build_model_XX(...)`). You can find the models implemented by us in
the file `./training/models.py`.

You can call the training script like this:

```bash
python train_classifier.py <Unique name / ID of experiment> head-only <number of ray workers>
```

Instead of calling the script directly, you can also use the slurm script files
contained in `./training/submit_scripts_slurm/` to perform the training runs. You
can use `submit_head_only.sh` to run an experiment on a single node containing
one or more GPUs.

However, to obtain reasonable training times, we recommend using additional
compute nodes to generate synthetic crystals and simulate their powder diffractogram. Depending on the model size, the number of needed cores to not
throttle the training process changes (bigger models train slower and need less
compute cores). You can use the script `submit.sh` (execute with `bash`, not
`sbatch`) to automatically spawn three slurm jobs on different compute nodes:
one head job and two compute worker jobs. The three jobs will wait until all
jobs are started and then initiate the training experiment. If your cluster
supports heterogeneous jobs, feel free to adapt the scripts accordingly.

Make sure to adapt all submit scripts to the exact specifications of your
cluster and change the name of the anaconda environment and potentially the path
to your `.bashrc` file in all submit scripts.

Each training experiment will put its data (TensorBoard data, logs, checkpoint files)
in a separate run directory. The current run directory will be printed in the beginning
of the training script.

The easiest way to track the progress and results of your training runs is to use
`TensorBoard`. Simply navigate to the run directory in your terminal and execute
`tensorboard --logdir .`.

There are several metrics that are logged to TensorBoard during a run:
- `accuracy/loss all`: Performance on ICSD test dataset
- `accuracy/loss match`: Performance on ICSD test dataset, only using structures that match
the simulation parameters (volume < 7000 angstroms, less than 100 atoms in asymmetric unit)
- `accuracy/loss random`: Performance on pXRDs from synthetically generated crystals
(same distribution as training data)
- `accuracy/loss match_correct_spgs`: Performance on ICSD test dataset, only using structures
that match the simulation parameters. Furthermore, the space group labels obtained using 
`spglib` are used instead of those provided by the ICSD.
- `accuracy/loss match_correct_spgs_pure`: Performance on ICSD test dataset, only using structures
that match the simulation parameters. Furthermore, the space group labels obtained using 
`spglib` are used instead of those provided by the ICSD. Also, only structures without partial
occupancies are used.
- `accuracy gap`: `accuracy random - accuracy match`

Additionally to those metrics, after each epoch, the current learning rate and the current
size of the `ray` queue object (indicating if enough workers are used) are logged.

## Inference

You can either use one of the models provided in our [latest release](https://github.com/aimat-lab/ML4pXRDs/releases/tag/v1.0)
or your own trained models to run inference on new diffractograms.

```python
import tensorflow.keras as keras

model = keras.models.load_model("path/to/your/model")

predictions = model.predict(your_diffractograms, batch_size=145)

```