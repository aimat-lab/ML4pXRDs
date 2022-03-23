import tensorflow.keras as keras
from dataset_simulations.random_simulation_utils import load_dataset_info
import numpy as np
import os
from dataset_simulations.simulation import Simulation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import sys
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

spgs = [2, 15]

validation_max_NO_wyckoffs = 100
validation_max_volume = 7000

model_path = (
    "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/initial_tests/17-03-2022_10-11-11/"
    + "final"
)

model = keras.models.load_model(model_path)

(
    probability_per_spg_per_element,
    probability_per_spg_per_element_per_wyckoff,
    NO_wyckoffs_prob_per_spg,
    corrected_labels,
    files_to_use_for_test_set,
    represented_spgs,
    NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element,
    denseness_factors_density_per_spg,
) = load_dataset_info()

path_to_patterns = "../../dataset_simulations/patterns/icsd_vecsei/"
jobid = os.getenv("SLURM_JOB_ID")
if jobid is not None and jobid != "":
    icsd_sim = Simulation(
        os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
        os.path.expanduser("~/Databases/ICSD/cif/"),
    )
    icsd_sim.output_dir = path_to_patterns
else:  # local
    icsd_sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )
    icsd_sim.output_dir = path_to_patterns

# icsd_sim.load(start=0, stop=files_to_use_for_test_set) # TODO: Change back
icsd_sim.load(start=0, stop=5)

n_patterns_per_crystal = len(icsd_sim.sim_patterns[0])

icsd_patterns_all = icsd_sim.sim_patterns
icsd_labels_all = icsd_sim.sim_labels
icsd_variations_all = icsd_sim.sim_variations
icsd_crystals_all = icsd_sim.sim_crystals
icsd_metas_all = icsd_sim.sim_metas

# Mainly to make the volume constraints correct:
conventional_errors_counter = 0
print("Calculating conventional structures...")
for i in reversed(range(0, len(icsd_crystals_all))):

    try:
        current_struc = icsd_crystals_all[i]
        analyzer = SpacegroupAnalyzer(current_struc)
        conv = analyzer.get_conventional_standard_structure()
        icsd_crystals_all[i] = conv

    except Exception as ex:

        print("Error calculating conventional cell of ICSD:")
        print(ex)
        conventional_errors_counter += 1

print(
    f"{conventional_errors_counter} of {len(icsd_crystals_all)} failed to convert to conventional cell."
)

for i in reversed(range(0, len(icsd_patterns_all))):

    if np.any(np.isnan(icsd_variations_all[i][0])) or icsd_labels_all[i][0] not in spgs:
        del icsd_patterns_all[i]
        del icsd_labels_all[i]
        del icsd_variations_all[i]
        del icsd_crystals_all[i]
        del icsd_metas_all[i]

# patterns that fall into the simulation parameter range (volume and NO_wyckoffs)
icsd_patterns_match = icsd_patterns_all.copy()
icsd_labels_match = icsd_labels_all.copy()
icsd_variations_match = icsd_variations_all.copy()
icsd_crystals_match = icsd_crystals_all.copy()
icsd_metas_match = icsd_metas_all.copy()

NO_wyckoffs_cached = {}
for i in reversed(range(0, len(icsd_patterns_match))):

    if validation_max_NO_wyckoffs is not None:
        is_pure, NO_wyckoffs, _, _, _ = icsd_sim.get_wyckoff_info(
            icsd_metas_match[i][0]
        )

        if icsd_metas_match[i][0] not in NO_wyckoffs_cached.keys():
            NO_wyckoffs_cached[icsd_metas_match[i][0]] = is_pure, NO_wyckoffs

    if (
        validation_max_volume is not None
        and icsd_crystals_match[i].volume > validation_max_volume
    ) or (
        validation_max_NO_wyckoffs is not None
        and NO_wyckoffs > validation_max_NO_wyckoffs
    ):
        del icsd_patterns_match[i]
        del icsd_labels_match[i]
        del icsd_variations_match[i]
        del icsd_crystals_match[i]
        del icsd_metas_match[i]

icsd_patterns_match_inorganic = icsd_patterns_match.copy()
icsd_labels_match_inorganic = icsd_labels_match.copy()
icsd_variations_match_inorganic = icsd_variations_match.copy()
icsd_crystals_match_inorganic = icsd_crystals_match.copy()
icsd_metas_match_inorganic = icsd_metas_match.copy()

exp_inorganic, exp_metalorganic, theoretical = icsd_sim.get_content_types()

for i in reversed(range(0, len(icsd_patterns_match_inorganic))):

    if icsd_metas_match_inorganic[i][0] not in exp_inorganic:

        del icsd_patterns_match_inorganic[i]
        del icsd_labels_match_inorganic[i]
        del icsd_variations_match_inorganic[i]
        del icsd_crystals_match_inorganic[i]
        del icsd_metas_match_inorganic[i]

print(f"Samples in match: {len(icsd_labels_match)}")
print(f"Samples in match_inorganic: {len(icsd_labels_match_inorganic)}")

val_y_match_inorganic = []
for i, label in enumerate(icsd_labels_match_inorganic):
    val_y_match_inorganic.extend([spgs.index(label[0])] * n_patterns_per_crystal)
val_y_match_inorganic = np.array(val_y_match_inorganic)

val_x_match_inorganic = []
for pattern in icsd_patterns_match_inorganic:
    for sub_pattern in pattern:
        val_x_match_inorganic.append(sub_pattern)

val_x_match_inorganic = np.expand_dims(val_x_match_inorganic, axis=2)

# gather metric names form model
metric_names = [metric.name for metric in model.metrics]

scores_match_inorganic = model.evaluate(
    x=val_x_match_inorganic, y=val_y_match_inorganic, verbose=0
)

assert metric_names[0] == "loss"

print(f"Accuracy: {scores_match_inorganic[1]}")

# find max loss structures and print pngs of them

losses = []
structures = []
for i, pattern in enumerate(icsd_patterns_match):
    for subpattern in pattern:
        x = subpattern[np.newaxis, :, np.newaxis]
        y = np.array([spgs.index(icsd_labels_match[i][0])])
        scores = model.evaluate(
            x=x,
            y=y,
            verbose=0,
        )

        loss = scores[0]

        losses.append(loss)
        structures.append(icsd_crystals_match[i])

        break

losses_structures_sorted = sorted(zip(losses, structures), key=lambda x: -1 * x[0])

os.system("mkdir -p ./max_losses")

counter = 0
for i, loss_structure in enumerate(losses_structures_sorted):

    if counter == 20:
        break

    if loss_structure[1].is_ordered:
        ase_struc = AseAtomsAdaptor.get_atoms(loss_structure[1])
        write(
            "./max_losses/" + str(loss_structure[0]) + ".png",
            ase_struc,
        )
        counter += 1
