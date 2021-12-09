# Author: Sakib Mahmud
# Source: https://github.com/Sakib1263/UNet-Segmentation-AutoEncoder-1D-2D-Tensorflow-Keras
# MIT License


# Import Necessary Libraries
import keras as k


def Conv_Block(inputs, model_width, kernel, multiplier):
    # 1D Convolutional Block
    x = k.layers.Conv1D(model_width * multiplier, kernel, padding="same")(inputs)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)

    return x


def trans_conv1D(inputs, model_width, multiplier):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = k.layers.Conv1DTranspose(
        model_width * multiplier, 2, strides=2, padding="same"
    )(
        inputs
    )  # Stride = 2, Kernel Size = 2
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)

    return x


def Concat_Block(input1, *argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = k.layers.concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs):
    # 1D UpSampling Block
    up = k.layers.UpSampling1D(size=2)(inputs)

    return up


def Feature_Extraction_Block(inputs, model_width, Dim2, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    latent = k.layers.Flatten()(inputs)
    latent = k.layers.Dense(feature_number, name="features")(latent)
    latent = k.layers.Dense(model_width * Dim2)(latent)
    latent = k.layers.Reshape((Dim2, model_width))(latent)

    return latent


def MultiResBlock(inputs, model_width, kernel, multiplier, alpha):
    # MultiRes Block
    # U {int} -- Number of filters in a corrsponding UNet stage
    # inp {keras layer} -- input layer

    w = alpha * model_width

    shortcut = inputs
    shortcut = Conv_Block(
        shortcut, int(w * 0.167) + int(w * 0.333) + int(w * 0.5), 1, multiplier
    )

    conv3x3 = Conv_Block(inputs, int(w * 0.167), kernel, multiplier)
    conv5x5 = Conv_Block(conv3x3, int(w * 0.333), kernel, multiplier)
    conv7x7 = Conv_Block(conv5x5, int(w * 0.5), kernel, multiplier)

    out = k.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
    out = k.layers.BatchNormalization()(out)
    out = k.layers.Add()([shortcut, out])
    out = k.layers.Activation("relu")(out)
    out = k.layers.BatchNormalization()(out)

    return out


def ResPath(inputs, model_depth, model_width, kernel, multiplier):
    # ResPath
    # filters {int} -- [description]
    # length {int} -- length of ResPath
    # inp {keras layer} -- input layer

    shortcut = inputs
    shortcut = Conv_Block(shortcut, model_width, 1, multiplier)

    out = Conv_Block(inputs, model_width, kernel, multiplier)
    out = k.layers.Add()([shortcut, out])
    out = k.layers.Activation("relu")(out)
    out = k.layers.BatchNormalization()(out)

    for i in range(1, model_depth):
        shortcut = out
        shortcut = Conv_Block(shortcut, model_width, 1, multiplier)

        out = Conv_Block(out, model_width, kernel, multiplier)
        out = k.layers.Add()([shortcut, out])
        out = k.layers.Activation("relu")(out)
        out = k.layers.BatchNormalization()(out)

    return out


def get_model():
    length = 2672
    model_depth = 4  # height
    num_channel = 1  # input
    model_width = 5  # first conv number of channels, danach immer verdoppeln
    kernel_size = 64
    output_nums = 1
    problem_type = "Regression"

    A_E = 0
    D_S = 0
    feature_number = 1024
    is_transconv = True

    # Variable UNet++ Model Design
    if (
        length == 0
        or model_depth == 0
        or model_width == 0
        or num_channel == 0
        or kernel_size == 0
    ):
        raise ValueError("Please Check the Values of the Input Parameters!")

    convs = {}
    levels = []

    # Encoding
    inputs = k.Input((length, num_channel))
    pool = inputs

    for i in range(1, (model_depth + 1)):
        conv = Conv_Block(pool, model_width, kernel_size, 2 ** (i - 1))
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** (i - 1))
        pool = k.layers.MaxPooling1D(pool_size=2)(conv)
        convs["conv%s" % i] = conv

    if A_E == 1:
        # Collect Latent Features or Embeddings from AutoEncoders
        pool = Feature_Extraction_Block(
            pool, model_width, int(length / (2 ** model_depth)), feature_number,
        )
    conv = Conv_Block(pool, model_width, kernel_size, 2 ** model_depth)
    conv = Conv_Block(conv, model_width, kernel_size, 2 ** model_depth)

    # Decoding
    convs_list = list(convs.values())
    if D_S == 1:
        level = k.layers.Conv1D(1, 1, name=f"level{model_depth}")(convs_list[0])
        levels.append(level)

    deconv = []
    deconvs = {}

    for i in range(1, (model_depth + 1)):
        for j in range(0, (model_depth - i + 1)):
            if (i == 1) and (j == (model_depth - 1)):
                if is_transconv:
                    deconv = Concat_Block(
                        convs_list[j], trans_conv1D(conv, model_width, 2 ** j)
                    )
                elif not is_transconv:
                    deconv = Concat_Block(convs_list[j], upConv_Block(conv))
                deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** j)
                deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** j)
                deconvs["deconv%s%s" % (j, i)] = deconv
            elif (i == 1) and (j < (model_depth - 1)):
                if is_transconv:
                    deconv = Concat_Block(
                        convs_list[j],
                        trans_conv1D(convs_list[j + 1], model_width, 2 ** j),
                    )
                elif not is_transconv:
                    deconv = Concat_Block(
                        convs_list[j], upConv_Block(convs_list[j + 1])
                    )
                deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** j)
                deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** j)
                deconvs["deconv%s%s" % (j, i)] = deconv
            elif i > 1:
                deconv_tot = deconvs["deconv%s%s" % (j, 1)]
                for ki in range(2, i):
                    deconv_temp = deconvs["deconv%s%s" % (j, ki)]
                    deconv_tot = Concat_Block(deconv_tot, deconv_temp)
                if is_transconv:
                    deconv = Concat_Block(
                        convs_list[j],
                        deconv_tot,
                        trans_conv1D(
                            deconvs["deconv%s%s" % ((j + 1), (i - 1))],
                            model_width,
                            2 ** j,
                        ),
                    )
                elif not is_transconv:
                    deconv = Concat_Block(
                        convs_list[j],
                        deconv_tot,
                        upConv_Block(deconvs["deconv%s%s" % ((j + 1), (i - 1))]),
                    )
                deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** j)
                deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** j)
                deconvs["deconv%s%s" % (j, i)] = deconv
            if (D_S == 1) and (j == 0) and (i < model_depth):
                level = k.layers.Conv1D(1, 1, name=f"level{model_depth - i}")(
                    deconvs["deconv%s%s" % (j, i)]
                )
                levels.append(level)

    deconv = deconvs["deconv%s%s" % (0, model_depth)]

    # Output
    outputs = []
    if problem_type == "Classification":
        outputs = k.layers.Conv1D(output_nums, 1, activation="softmax", name="out")(
            deconv
        )
    elif problem_type == "Regression":
        outputs = k.layers.Conv1D(output_nums, 1, activation="linear", name="out")(
            deconv
        )

    model = k.Model(inputs=[inputs], outputs=[outputs])

    if D_S == 1:
        levels.append(outputs)
        levels.reverse()
        model = k.Model(inputs=[inputs], outputs=levels)

    return model
