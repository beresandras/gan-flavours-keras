import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


class AdaptiveAugmenter(keras.Model):
    def __init__(self, target_accuracy, integration_steps, input_shape):
        super().__init__()

        self.probability = tf.Variable(0.0)

        self.target_accuracy = target_accuracy
        self.integration_steps = integration_steps

        max_translation = 0.125
        max_rotation = 0.125
        max_zoom = 0.25

        self.augmenter = keras.Sequential(
            [
                layers.InputLayer(input_shape=input_shape),
                preprocessing.RandomFlip("horizontal"),
                preprocessing.RandomTranslation(
                    height_factor=max_translation,
                    width_factor=max_translation,
                    interpolation="nearest",
                ),
                preprocessing.RandomRotation(factor=max_rotation),
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

            augmentation_values = tf.random.uniform(
                shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
            )
            augmentation_bools = tf.math.less(augmentation_values, self.probability)

            images = tf.where(augmentation_bools, augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = tf.reduce_mean(0.5 * (1.0 + tf.sign(real_logits)))
        # current_accuracy = tf.reduce_mean(tf.cast(tf.math.greater(real_logits, 0.0), dtype=tf.float32))
        # current_accuracy = tf.reduce_mean(tf.keras.activations.sigmoid(real_logits))

        self.probability.assign(
            tf.clip_by_value(
                self.probability
                + (current_accuracy - self.target_accuracy) / self.integration_steps,
                0,
                1,
            )
        )