import sys

sys.path.append("../dataset_simulations/")
sys.path.append("../")

import tensorflow.keras as keras
from dataset_simulations.core.quick_simulation import get_random_xy_patterns
import numpy as np
from models import build_model_park
from datetime import datetime
import os
from sklearn.utils import shuffle

# TODO: Add validation set (ICSD data)

tag = "debug"

test_every_X_epochs = 10
batches_per_epoch = 1500
max_NO_elements = 10
structures_per_spg = 64
# => 96k structures per spg per epoch

#spgs = [1, 14]
spgs = [14, 104]

# like in the Vecsei paper:
start_angle, end_angle, N = 5, 90, 8501
angle_range = np.linspace(start_angle, end_angle, N)

NO_workers = 8
queue_size = 20
do_multi_processing = True

out_base = (
    "classifier_spgs/" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + "_" + tag + "/"
)
os.system("mkdir -p " + out_base)


class CustomSequence(keras.utils.Sequence):
    def __init__(self, number_of_batches):
        self.number_of_batches = number_of_batches

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):

        patterns, labels = get_random_xy_patterns(
            spgs=spgs,
            structures_per_spg=structures_per_spg,
            wavelength=1.5406,  # TODO: Cu-K line, when testing on ICSD data, switch to 1.207930 wavelength (scaling)
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


sequence = CustomSequence(batches_per_epoch)

model = build_model_park(None, N, len(spgs))

model.fit(
    x=sequence,
    epochs=test_every_X_epochs,
    batch_size=batches_per_epoch,
    # validation_data=(x_val, y_val), # TODO: Can I use this for testing on the ICSD data?
    # Yes, I can; use validation_freq = N_epochs
    callbacks=[keras.callbacks.TensorBoard(out_base + "tuner_tb")],
    verbose=1,
    workers=NO_workers,
    max_queue_size=queue_size,
    use_multiprocessing=do_multi_processing,
    steps_per_epoch=batches_per_epoch,
)

model.save(out_base + "final")
