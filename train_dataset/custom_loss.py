# Source: https://github.com/j-bernardi/bayesian-label-smoothing/blob/main/losses/custom_loss.py

import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

tf.keras.backend.set_floatx("float32")


class CustomSmoothedWeightedCCE:
    """Custom CCE implementing label smoothing, class weighting
    Categorical Crossentropy loss function with custom
    implementation for label smoothing and 
    """

    def __init__(self, class_weights=None):
        self.class_weights_array = class_weights

        if class_weights is None or all(v == 1.0 for v in class_weights):
            self.class_weights = None
        else:
            self.class_weights = tf.convert_to_tensor(class_weights, dtype="float32")

        self.unreduced_cce_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=losses_utils.ReductionV2.NONE,
        )

    def __call__(self, y_true, y_pred, sample_weight=None):

        if sample_weight is not None:
            raise NotImplementedError(
                f"This loss function is only for implementing class weights."
                f" sample_weight not None ({sample_weight}) not valid."
            )

        unreduced_loss = tf.dtypes.cast(
            self.unreduced_cce_fn(y_true, y_pred), "float32"
        )

        # Weight the losses
        if self.class_weights is not None:
            true_classes = tf.argmax(y_true, axis=-1)
            weight_mask = tf.gather(self.class_weights, true_classes)
            loss = tf.math.multiply(unreduced_loss, weight_mask)
        else:
            loss = unreduced_loss

        return losses_utils.reduce_weighted_loss(
            loss, reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
        )

    def get_config(self):
        return {"class_weights": self.class_weights_array}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
