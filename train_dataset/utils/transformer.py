# Base is from https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3
# Time2vec has been replaced with trainable embeddings here, similar to https://github.com/adonis1022/time_series_Transformer/blob/master/TrainablePositionalEmbeddings.py
# and https://keras.io/examples/vision/image_classification_with_vision_transformer/

# TODO: Maybe rather use d-space instead of angle, because then periodic positional embeddings make more sense.

import tensorflow.keras as keras
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras import backend as K
import numpy as np

class TransformerPositionalEmbedding(keras.Model):
    """
    Trainable positional embeddings: to be added to the inputs of Transformer block to learn 
    sequence information carried by the sentences.
    """

    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)

        self.input_embedding_width = width
        self.embedding = keras.layers.Dense(self.input_embedding_width)

    def build(self, input_shape):

        sequence_length = input_shape[-2]

        self.position_embedding = self.add_weight(
            shape=(sequence_length, self.input_embedding_width),
            initializer='uniform',
            name='position_embeddings',
            trainable=True)

    def call(self, inputs, **kwargs):
        x = keras.layers.TimeDistributed(self.embedding)(inputs) # embed 1D sequence into self.width-D sequence (time-independently)
        return x + self.position_embedding

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x

class ModelTrunk(keras.Model):
    def __init__(self, input_shape, name='ModelTrunk', num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, input_embedding_width=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_embedding = TransformerPositionalEmbedding(input_embedding_width)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
        self.shape = input_shape
        
    def call(self, inputs):

        x = self.pos_embedding(inputs)

        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2])) # flat vector of features out

def get_transformer():

    model = ModelTrunk((None,7,1), num_heads=2, head_size=128)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ],  # here from_logits is not needed, since argmax will be the same
    )

    return model

if __name__ == "__main__":

    model = get_transformer()

    print(model.predict(np.expand_dims(np.array([[1.5,5.3,2.5,4.1,5.8,2.1,5.7]], dtype=float), -1)))

    # Why do we call MultiHeadAttention only with key, value?
    #query = inputs[0]
    #key = inputs[1]
    #value = inputs[2] if len(inputs) > 2 else key