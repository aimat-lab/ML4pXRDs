import sys

#sys.path.append("../dataset_simulations/")
#sys.path.append("../")

import tensorflow.keras as keras
from dataset_simulations.core.quick_simulation import get_random_xy_patterns
import numpy as np
from models import build_model_park
from datetime import datetime
import os
from sklearn.utils import shuffle
from dataset_simulations.simulation import Simulation
from scipy import interpolate as ip

import ray
from ray.util.queue import Queue


tag = "debug"

#test_every_X_epochs = 10
test_every_X_epochs = 1
batches_per_epoch = 1500
NO_epochs = 100

max_NO_elements = 10
structures_per_spg = 64
# => 96k structures per spg per epoch

#spgs = [14, 104] # works well, relatively high val_acc
#spgs = [129, 176] # 93.15%, pretty damn well!
spgs = [2, 15] # pretty much doesn't work at all (so far!), val_acc ~40%, after a full night: ~43%
# after a full night with random volume factors: binary_accuracy: 0.7603 - val_loss: 0.8687 - val_binary_accuracy: 0.4749; still bad

# like in the Vecsei paper:
start_angle, end_angle, N = 5, 90, 8501
start_angle, end_angle, N = 360/(2*np.pi)*np.arcsin(1.207930/1.5406 * np.sin(2*np.pi/360*start_angle)), 360/(2*np.pi)*np.arcsin(1.207930/1.5406 * np.sin(2*np.pi/360*end_angle)), 8501 # until ICSD has not been re-simulated with Cu-K line
angle_range = np.linspace(start_angle, end_angle, N)
print(f"Start-angle: {start_angle}, end-angle: {end_angle}")

NO_workers = 8
queue_size = 200
queue_size_tf = 100

out_base = (
    "classifier_spgs/" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + "_" + tag + "/"
)
os.system("mkdir -p " + out_base)

# load validation set (ICSD):

number_of_values_initial = 9018
simulated_range = np.linspace(0, 90, number_of_values_initial)
start_index = np.argwhere(simulated_range >= start_angle)[0][0]
end_index = np.argwhere(simulated_range <= end_angle)[-1][0]
used_range = simulated_range[start_index : end_index + 1]

path_to_patterns = "../dataset_simulations/patterns/icsd/"
jobid = os.getenv("SLURM_JOB_ID")
if jobid is not None and jobid != "":
    icsd_sim = Simulation(
        "/home/ws/uvgnh/Databases/ICSD/ICSD_data_from_API.csv",
        "/home/ws/uvgnh/Databases/ICSD/cif/",
    )
    icsd_sim.output_dir = path_to_patterns
else:
    icsd_sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )
    icsd_sim.output_dir = path_to_patterns

icsd_sim.load(load_only=5)

n_patterns_per_crystal = len(icsd_sim.sim_patterns[0])

icsd_patterns = icsd_sim.sim_patterns
icsd_labels = icsd_sim.sim_labels
icsd_variations = icsd_sim.sim_variations
icsd_crystals = icsd_sim.sim_crystals

"""
dist_y = []
for i, label in enumerate(icsd_labels):
    dist_y.append(label[0])
print(np.bincount(dist_y))
"""

for i in reversed(range(0, len(icsd_patterns))):
    if (
        np.any(np.isnan(icsd_variations[i][0]))
        or icsd_labels[i][0] not in spgs
    ):
        del icsd_patterns[i]
        del icsd_labels[i]
        del icsd_variations[i]
        del icsd_crystals[i]

val_y = []
for i, label in enumerate(icsd_labels):
    val_y.extend([spgs.index(label[0])] * n_patterns_per_crystal)
val_y = np.array(val_y)

for i in range(0, len(spgs)):
    print(f"Spg {spgs[i]} : {np.sum(val_y==i)}")

val_x = []
for pattern in icsd_patterns:
    for sub_pattern in pattern:

        f = ip.CubicSpline(used_range, sub_pattern[start_index: end_index+1], bc_type="natural")
        actual_val = f(angle_range)

        val_x.append(actual_val)

assert not np.any(np.isnan(val_x))
assert not np.any(np.isnan(val_y))
assert len(val_x) == len(val_y)

val_x = np.expand_dims(val_x, axis=2)

ray.init(include_dashboard=True, num_cpus=NO_workers)

queue = Queue(maxsize=queue_size) # store a maximum of `queue_size` batches

@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator(queue, spgs, structures_per_spg, N, start_angle, end_angle, max_NO_elements):

    while True:

        try:

            patterns, labels = get_random_xy_patterns(
                    spgs=spgs,
                    structures_per_spg=structures_per_spg,
                    #wavelength=1.5406,  # TODO: Cu-K line, when testing on ICSD data, switch to 1.207930 wavelength (scaling)
                    wavelength=1.207930, # until ICSD has not been re-simulated with Cu-K line
                    N=N,
                    two_theta_range=(start_angle, end_angle),
                    max_NO_elements=max_NO_elements,
                    do_print=False,
                )

            patterns, labels = shuffle(patterns, labels)

            # Set the label to the right index:
            for i in range(0, len(labels)):
                labels[i] = spgs.index(labels[i])

            patterns = np.array(patterns)
            patterns = np.expand_dims(patterns, axis=2)

            labels = np.array(labels)

            queue.put((patterns,labels)) # blocks if queue is full, which is good
        
        except Exception as ex:

            print("Error occurred in worker:")
            print(ex)

# Start worker tasks
for i in range(0, NO_workers):
    batch_generator.remote(queue, spgs, structures_per_spg, N, start_angle, end_angle, max_NO_elements)

class CustomSequence(keras.utils.Sequence):
    def __init__(self, number_of_batches):
        self.number_of_batches = number_of_batches

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):

        """
        patterns, labels = get_random_xy_patterns(
            spgs=spgs,
            structures_per_spg=structures_per_spg,
            #wavelength=1.5406,  # TODO: Cu-K line, when testing on ICSD data, switch to 1.207930 wavelength (scaling)
            wavelength=1.207930, # until ICSD has not been re-simulated with Cu-K line
            N=N,
            two_theta_range=(start_angle, end_angle),
            max_NO_elements=max_NO_elements,
            do_print=False,
        )

        patterns, labels = shuffle(patterns, labels)

        # Set the label to the right index:
        for i in range(0, len(labels)):
            labels[i] = spgs.index(labels[i])

        patterns = np.array(patterns)
        patterns = np.expand_dims(patterns, axis=2)

        labels = np.array(labels)

        return (
            patterns, labels
        )
        """

        return queue.get()


sequence = CustomSequence(batches_per_epoch)

model = build_model_park(None, N, len(spgs))

model.fit(
    x=sequence,
    epochs=NO_epochs,
    batch_size=batches_per_epoch,
    validation_data=(val_x, val_y),
    validation_freq=test_every_X_epochs,
    callbacks=[keras.callbacks.TensorBoard(out_base + "tuner_tb")],
    verbose=1,
    workers=1,
    max_queue_size=queue_size_tf,
    use_multiprocessing=False,
    steps_per_epoch=batches_per_epoch,
)

model.save(out_base + "final")

ray.shutdown()