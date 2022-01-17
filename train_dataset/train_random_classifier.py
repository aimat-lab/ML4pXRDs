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
import pickle

tag = "just_test_it"

test_every_X_epochs = 1
batches_per_epoch = 150
NO_epochs = 1000

structures_per_spg = 1
NO_corn_sizes = 5
# => batch size: 1*5*200=~1000

NO_workers = 126+14
queue_size = 200
queue_size_tf = 100

compare_distributions = True
NO_random_batches = 20

max_NO_elements = 10

verbosity = 2

local = False
if local:
    structures_per_spg = 1 # decrease batch size
    NO_workers = 8
    verbosity = 1

#spgs = [14, 104] # works well, relatively high val_acc
#spgs = [129, 176] # 93.15%, pretty damn well!
#spgs = [2, 15] # pretty much doesn't work at all (so far!), val_acc ~40%, after a full night: ~43%
# after a full night with random volume factors: binary_accuracy: 0.7603 - val_loss: 0.8687 - val_binary_accuracy: 0.4749; still bad
#spgs = [14,104,129,176] # after 100 epochs: 0.8503 val accuracy
# all spgs (~200): loss: sparse_categorical_accuracy: 0.1248 - val_sparse_categorical_accuracy: 0.0713; it is a beginning!

# like in the Vecsei paper:
start_angle, end_angle, N = 5, 90, 8501
start_angle, end_angle, N = 360/(2*np.pi)*np.arcsin(1.207930/1.5406 * np.sin(2*np.pi/360*start_angle)), 360/(2*np.pi)*np.arcsin(1.207930/1.5406 * np.sin(2*np.pi/360*end_angle)), 8501 # until ICSD has not been re-simulated with Cu-K line
angle_range = np.linspace(start_angle, end_angle, N)
print(f"Start-angle: {start_angle}, end-angle: {end_angle}")

out_base = (
    "classifier_spgs/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_" + tag + "/"
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

icsd_sim.load(load_only=5 if not local else 1)

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

spgs = sorted(np.unique([item[0] for item in icsd_labels]))

for i in reversed(range(0, len(icsd_patterns))):
    if (
        np.any(np.isnan(icsd_variations[i][0]))
        or icsd_labels[i][0] not in spgs
    ):
        del icsd_patterns[i]
        del icsd_labels[i]
        del icsd_variations[i]
        del icsd_crystals[i]

if compare_distributions:

    with open(out_base + "spgs.pickle", "wb") as file:
        pickle.dump(spgs, file)

    with open(out_base + "icsd_data.pickle", "wb") as file:
        pickle.dump((icsd_crystals, icsd_labels, [item[0] for item in icsd_variations]), file)

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

if not local:
    ray.init(address='auto', include_dashboard=False)
    #ray.init(include_dashboard=True, num_cpus=NO_workers)
else:
    ray.init(include_dashboard=False)

print()
print(ray.cluster_resources())
print()

queue = Queue(maxsize=queue_size) # store a maximum of `queue_size` batches

@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator_with_additional(spgs, structures_per_spg, N, start_angle, end_angle, max_NO_elements, NO_corn_sizes):

    patterns, labels, structures, corn_sizes = get_random_xy_patterns(
                spgs=spgs,
                structures_per_spg=structures_per_spg,
                #wavelength=1.5406,  # TODO: Cu-K line
                wavelength=1.207930, # until ICSD has not been re-simulated with Cu-K line
                N=N,
                NO_corn_sizes=NO_corn_sizes,
                two_theta_range=(start_angle, end_angle),
                max_NO_elements=max_NO_elements,
                do_print=False,
                return_additional=True
            )

    # Set the label to the right index:
    for i in range(0, len(labels)):
        labels[i] = spgs.index(labels[i])

    patterns = np.array(patterns)
    labels = np.array(labels)

    return patterns, labels, structures, corn_sizes

@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator_queue(queue, spgs, structures_per_spg, N, start_angle, end_angle, max_NO_elements, NO_corn_sizes):

    while True:

        try:

            patterns, labels = get_random_xy_patterns(
                spgs=spgs,
                structures_per_spg=structures_per_spg,
                #wavelength=1.5406,  # TODO: Cu-K line
                wavelength=1.207930, # until ICSD has not been re-simulated with Cu-K line
                N=N,
                NO_corn_sizes=NO_corn_sizes,
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
            print(
                type(ex).__name__,          # TypeError
                __file__,                  # /tmp/example.py
                ex.__traceback__.tb_lineno  # 2
            )

if compare_distributions:
    # pre-store some batches to compare to the rightly / falsely classified icsd samples

    random_comparison_crystals = []
    random_comparison_labels = []
    random_comparison_corn_sizes = []

    object_refs = []
    for i in range(NO_random_batches):
        ref = batch_generator_with_additional.remote(spgs, 1, N, start_angle, end_angle, max_NO_elements, 1)
        object_refs.append(ref)

    results = ray.get(object_refs)

    for result in results:
        patterns, labels, crystals, corn_sizes = result
        random_comparison_crystals.extend(crystals)
        random_comparison_labels.extend(labels)
        random_comparison_corn_sizes.extend(corn_sizes)

    with open(out_base + "random_data.pickle", "wb") as file:
        pickle.dump((random_comparison_crystals, random_comparison_labels, random_comparison_corn_sizes), file)

# Start worker tasks
for i in range(0, NO_workers):
    batch_generator_queue.remote(queue, spgs, structures_per_spg, N, start_angle, end_angle, max_NO_elements, NO_corn_sizes)

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
    verbose=verbosity,
    workers=1,
    max_queue_size=queue_size_tf,
    use_multiprocessing=False,
    steps_per_epoch=batches_per_epoch,
)

model.save(out_base + "final")

prediction = model.predict(val_x)
prediction = np.argmax(prediction, axis=1)

#print(prediction)
#print(len(prediction))
#print(len(val_x))
#print(len(icsd_crystals))

rightly_indices = np.argwhere(prediction == val_y)[:, 0]
falsely_indices = np.argwhere(prediction != val_y)[:, 0]

#print(rightly_indices)
#print(falsely_indices)

with open(out_base + "rightly_falsely.pickle", "wb") as file:
    pickle.dump((rightly_indices, falsely_indices), file)

ray.shutdown()
