from UNet_1DCNN import UNet
import tensorflow.keras as keras
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import pickle
import os

mode = "removal"  # also possible: "info"

N = 9018
pattern_x = np.linspace(0, 90, N)

# only use a restricted range of the simulated patterns
start_x = 10
end_x = 50
start_index = np.argwhere(pattern_x >= start_x)[0][0]
end_index = np.argwhere(pattern_x <= end_x)[-1][0]
pattern_x = pattern_x[start_index : end_index + 1]
N = len(pattern_x)

my_unet = UNet(N, 3, 1, 5, 64, output_nums=1, problem_type="Regression")
model = my_unet.UNet()

# keras.utils.plot_model(model, show_shapes=True)

model.summary()


# Read training data in separate thread to limit memory usage
def read_training_data(Q):

    with open("../dataset_simulations/patterns/noise_background/data", "rb") as file:
        data = pickle.load(file)

    if mode == "removal":
        x = data[0][:, start_index : end_index + 1]
        y = data[1][:, start_index : end_index + 1]
    elif mode == "info":
        x = data[0][:, start_index : end_index + 1]
        y = data[2][:, start_index : end_index + 1]
    else:
        raise Exception("Mode not supported.")

    # Split into train, validation, test set + shuffle
    x, y = shuffle(x, y, random_state=1234)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    sc = StandardScaler()
    x_train = np.expand_dims(sc.fit_transform(x_train), axis=2)
    x_test = np.expand_dims(sc.transform(x_test), axis=2)
    x_val = np.expand_dims(sc.transform(x_val), axis=2)

    with open("unet/" + mode + "_cps/scaler", "wb") as file:
        pickle.dump(sc, file)

    Q.put((x_train, x_test, x_val, y_train, y_test, y_val))


Q = Queue()
process_1 = Process(target=read_training_data, args=[Q])
process_1.start()
(x_train, x_test, x_val, y_train, y_test, y_val) = Q.get()
process_1.join()

if mode == "removal":
    model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
elif mode == "info":
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
else:
    raise Exception("Mode not supported.")

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath="unet/" + mode + "_cps" + "/weights{epoch}",
    save_weights_only=True,
    verbose=1,
)
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=100,
    verbose=2,
    callbacks=[
        cp_callback,
        keras.callbacks.TensorBoard(log_dir="unet/" + mode + "_tb"),
    ],
    validation_data=(x_val, y_val),
)

if mode == "removal":
    predictions = model.predict(x_test[0:20])
else:
    probability_model = keras.Sequential([model, keras.layers.Activation("sigmoid")])
    predictions = probability_model.predict(x_test[0:20])

os.system("mkdir -p predictions")
for i, prediction in enumerate(predictions):
    if mode == "removal":
        plt.plot(pattern_x, prediction)
        plt.plot(pattern_x, x_test[i])
        plt.savefig(f"predictions/prediction_{i}.pdf")
    elif mode == "info":
        plt.scatter(pattern_x, prediction)
        plt.plot(pattern_x, x_test[i])
        plt.savefig(f"predictions/prediction_{i}.pdf")
    else:
        raise Exception("Mode not recognized.")
