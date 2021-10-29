from UNet_1DCNN import UNet
import tensorflow as tf
import tensorflow.keras as keras
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import gc
import time
from multiprocessing import Process, Queue


N = 4224
pattern_x = np.linspace(0, 90, N)

my_unet = UNet(N, 3, 1, 5, 64, output_nums=1, problem_type="Regression")
model = my_unet.UNet()

# keras.utils.plot_model(model, show_shapes=True)

model.summary()

# read training data:


def load_and_shuffle(Q):
    with open("../dataset_simulations/patterns/noise_background/data", "rb") as file:
        ys_altered, ys_unaltered = pickle.load(file)

    x = ys_altered
    y = ys_unaltered

    # Split into train, validation, test set + shuffle
    x, y = shuffle(x, y, random_state=1234)

    Q.put((x, y))


Q = Queue()
process_1 = Process(target=load_and_shuffle, args=[Q])
process_1.start()
print(Q.get())
process_1.join()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_val = sc.transform(x_val)

x = np.expand_dims(x, axis=2)
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_val = np.expand_dims(x_val, axis=2)

model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
model.fit(ys_altered, ys_unaltered, epochs=5, batch_size=5)
model.save_weights("weights")

predictions = model(x_test).numpy()

for i, prediction in enumerate(predictions):
    plt.plot(pattern_x, prediction)
    plt.plot(pattern_x, x_test[i])
    plt.show()
