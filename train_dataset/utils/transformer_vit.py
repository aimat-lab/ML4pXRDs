# From https://keras.io/examples/vision/image_classification_with_vision_transformer/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from train_dataset.utils.AdamWarmup import AdamWarmup
from train_dataset.utils.AdamWarmup import calc_train_steps
import numpy as np
from addons.tensorflow_addons.layers import MultiHeadAttention

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        #x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, inputs):

        #print(inputs.shape)

        images = inputs[:,:,None,:] # add a pseudo second dimensions to be able to use extract_patches function

        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        #print(patches.shape)

        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):

        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):

        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


"""
## Build the ViT model
The ViT model consists of multiple Transformer blocks,
which use the `layers.MultiHeadAttention` layer as a self-attention mechanism
applied to the sequence of patches. The Transformer blocks produce a
`[batch_size, num_patches, projection_dim]` tensor, which is processed via an
classifier head with softmax to produce the final class probabilities output.
Unlike the technique described in the [paper](https://arxiv.org/abs/2010.11929),
which prepends a learnable embedding to the sequence of encoded patches to serve
as the image representation, all the outputs of the final Transformer block are
reshaped with `layers.Flatten()` and used as the image
representation input to the classifier head.
Note that the `layers.GlobalAveragePooling1D` layer
could also be used instead to aggregate the outputs of the Transformer block,
especially when the number of patches and the projection dimensions are large.
"""

def build_model_transformer_vit(
    hp=None,
    number_of_input_values=8501,
    number_of_output_labels=2,
    lr=0.001,
    epochs=200,
    steps_per_epoch=1500,
):

    patch_size = 32
    num_patches = number_of_input_values // patch_size

    projection_dim = 64 
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 1 # 8
    mlp_head_units = [512, 512]  # Size of the dense layers of the final classifier # [2048, 1024]

    ##########

    inputs = layers.Input(shape=(8501,1))

    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        #attention_output = layers.MultiHeadAttention(
        #    num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        #)(x1, x1)

        print("##### ", x1.shape)

        attention_output = MultiHeadAttention(
            num_heads=num_heads, head_size=projection_dim #, dropout=0.1
        )([x1, x1])

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    #representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(number_of_output_labels)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)

    ##########

    model.summary()

    total_steps, warmup_steps = calc_train_steps(
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        warmup_proportion=0.1,
    )

    model.compile(
        optimizer=AdamWarmup(total_steps, warmup_steps, learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

if __name__ == "__main__":

    model = build_model_transformer_vit()
    print(model.predict(np.random.random(size=(1,8501,1))))