import numpy as np
import matplotlib.pyplot as plt
import sys
from UNet_1DCNN import UNet

sys.path.append("../")
from train_dataset.baseline_utils import load_experimental_data
from scipy import interpolate as ip
import tensorflow.keras as keras
import pandas as pd

from sklearn.metrics import confusion_matrix


remove_background = True

if __name__ == "__main__":

    classifier_model_name = "narrow_03-12-2021_12:31:23_test"
    unet_model_name = "removal_03-12-2021_16-48-30_UNetPP"
    classify_is_pure = False
    do_plot = False
    only_pure_exps = True

    xs_exp, ys_exp, ids_exp = load_experimental_data(
        "exp_data/XRDdata_classification.csv"
    )

    classifier_model = keras.models.load_model(
        "classifier/" + classifier_model_name + "/final", compile=False
    )
    unet_model = keras.models.load_model("unet/" + unet_model_name + "/final")

    data_true_labels = pd.read_csv(
        "exp_data/experimental_phases"
        + ("_only_pure" if only_pure_exps else "")
        + ".txt",
        delimiter=" ",
        skiprows=0,
        header=None,
    )
    sample_names = list(data_true_labels.iloc[:, 0])
    sample_ids = [int("".join(filter(str.isdigit, item))) for item in sample_names]

    spg_labels_pd = pd.read_csv(
        "exp_data/experimental_spgs"
        + ("_only_pure" if only_pure_exps else "")
        + ".txt",
        delimiter=" ",
        skiprows=0,
        header=None,
    )
    spg_labels = np.array(spg_labels_pd.iloc[:, 0])

    if only_pure_exps:
        # remove all non-pure samples from the test data

        i = len(ids_exp) - 1
        for id in reversed(ids_exp):
            if id not in sample_ids:
                xs_exp = np.delete(xs_exp, i, 1)
                ys_exp = np.delete(ys_exp, i, 1)
                del ids_exp[i]
            i -= 1

    correct_counter = 0

    predictions = []
    trues = []

    for i in range(0, xs_exp.shape[1]):

        current_xs = xs_exp[:, i]
        current_ys = ys_exp[:, i]

        if remove_background:

            current_xs = current_xs[0:2672]
            ys = current_ys[0:2672]
            ys -= np.min(ys)
            ys = ys / np.max(ys)

            if do_plot:
                plt.plot(current_xs, ys, label="Experimental rescaled")

            ys_to_be_corrected = np.expand_dims([ys], axis=2)
            corrected = unet_model.predict(ys_to_be_corrected)

            # with open("classifier/scaler", "rb") as file:
            #    sc = pickle.load(file)
            #    ys_to_be_classified = sc.transform(
            #        [corrected[0, :, 0] / np.max(corrected[0, :, 0])]
            #    )
            #    ys_to_be_classified = np.expand_dims(ys_to_be_classified, axis=2)

            # Scale experimental pattern to the right dimension
            number_of_values_initial = 9018
            simulated_range = np.linspace(0, 90, number_of_values_initial)
            start_x = 10
            end_x = 50
            step = 1
            start_index = np.argwhere(simulated_range >= start_x)[0][0]
            end_index = np.argwhere(simulated_range <= end_x)[-1][0]
            used_range = simulated_range[start_index : end_index + 1 : step]
            number_of_values = len(used_range)
            f = ip.CubicSpline(current_xs, corrected[0, :, 0], bc_type="natural")
            ys_to_be_classified = f(used_range)
            ys_to_be_classified = np.expand_dims([ys_to_be_classified], axis=2)

            softmax_activation = keras.layers.Activation("softmax")(
                classifier_model.get_layer("outputs_softmax").output
            )
            prob_model_softmax = keras.Model(
                inputs=classifier_model.layers[0].output, outputs=softmax_activation
            )
            prediction_softmax = prob_model_softmax.predict(ys_to_be_classified)
            prediction_softmax = np.argmax(prediction_softmax, axis=1)

            predictions.append(prediction_softmax[0])

            if classify_is_pure:
                sigmoid_activation = keras.layers.Activation("sigmoid")(
                    classifier_model.get_layer("output_sigmoid").output
                )
                prob_model_sigmoid = keras.Model(
                    inputs=classifier_model.layers[0].output, outputs=sigmoid_activation
                )
                prediction_sigmoid = prob_model_sigmoid.predict(ys_to_be_classified)
                prediction_sigmoid = prediction_sigmoid[:, 0]
                prediction_sigmoid = np.where(prediction_sigmoid > 0.5, 1, 0)

            narrow_phases = ["Fm-3m", "Ia-3", "P63/m"]

            trues.append(narrow_phases.index(spg_labels[i]))

            print(
                f"Output of phase classification: {narrow_phases[prediction_softmax[0]]} (True: {spg_labels[i]})"
            )

            if narrow_phases[prediction_softmax[0]] == spg_labels[i]:
                correct_counter += 1

            if spg_labels[i] not in narrow_phases:
                raise Exception("Found not-supported phase in csv file.")

            if classify_is_pure:
                purities = ["non-pure", "pure"]
                print(
                    f"Output of pure classification: {purities[prediction_sigmoid[0]]}"
                )

                if do_plot:
                    plt.plot(
                        current_xs,
                        corrected[0, :, 0],
                        label=f"Corrected via U-Net\n\nPredicted labels: {narrow_phases[prediction_softmax[0]]}, {purities[prediction_sigmoid[0]]}\nTrue label: {spg_labels[i]}",
                    )

            else:

                if do_plot:
                    plt.plot(
                        current_xs,
                        corrected[0, :, 0],
                        label=f"Corrected via U-Net\n\nPredicted labels: {narrow_phases[prediction_softmax[0]]}\nTrue label: {spg_labels[i]}",
                    )

            if do_plot:
                plt.plot(current_xs, np.zeros(len(current_xs)))

            if do_plot:
                plt.plot(
                    current_xs, ys - corrected[0, :, 0], label="Background and noise"
                )
                plt.legend()
                plt.show()

    print(f"{correct_counter / len(spg_labels)} prediction accurary")

    trues = np.array(trues)
    predictions = np.array(predictions)

    # print(f"{np.sum(trues==predictions) / len(trues)}")

    print(narrow_phases)
    print(confusion_matrix(trues, predictions))

    plt.figure()

    for i in range(0, 3):
        first_index = np.argwhere(trues == i)[1 if i < 2 else 0, 0]

        current_xs = xs_exp[:, first_index]
        current_ys = ys_exp[:, first_index]

        plt.plot(current_xs, current_ys, label=spg_labels[first_index])

    plt.xlabel(r"$2 \theta$")
    plt.ylabel(r"Intensity / arb. unit")
    plt.legend()
    plt.savefig("narrow_examples.png", dpi=300)
    plt.show()
