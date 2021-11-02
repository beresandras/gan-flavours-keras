import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from utils import step


# augments images with a probability that is dynamically updated during training
class AdaptiveAugmenter(keras.Model):
    def __init__(self, target_accuracy, integration_steps, input_shape):
        super().__init__()

        # stores the current probability of an image being augmented
        self.probability = tf.Variable(0.0)

        self.target_accuracy = target_accuracy
        self.integration_steps = integration_steps

        max_translation = 0.125
        max_rotation = 0.125
        max_zoom = 0.25

        # the corresponding augmentation names from the paper are shown above each layer
        self.augmenter = keras.Sequential(
            [
                layers.InputLayer(input_shape=input_shape),
                # blitting/x-flip:
                preprocessing.RandomFlip("horizontal"),
                # blitting/integer translation:
                preprocessing.RandomTranslation(
                    height_factor=max_translation,
                    width_factor=max_translation,
                    interpolation="nearest",
                ),
                # geometric/rotation:
                preprocessing.RandomRotation(factor=max_rotation),
                # geometric/isotropic and anisotropic scaling:
                preprocessing.RandomZoom(
                    height_factor=(-max_zoom, 0.0), width_factor=(-max_zoom, 0.0)
                ),
            ],
            name="adaptive_augmenter",
        )

    def call(self, images, training):
        batch_size = tf.shape(images)[0]

        if training:
            augmented_images = self.augmenter(images, training)

            # during training either the original or the augmented images are selected
            # based on self.probability
            augmentation_values = tf.random.uniform(
                shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
            )
            augmentation_bools = tf.math.less(augmentation_values, self.probability)

            images = tf.where(augmentation_bools, augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = tf.reduce_mean(step(real_logits))

        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - self.target_accuracy
        self.probability.assign(
            tf.clip_by_value(
                self.probability + accuracy_error / self.integration_steps, 0.0, 1.0
            )
        )