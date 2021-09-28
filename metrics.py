import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


class KID(keras.metrics.Metric):
    def __init__(self, name="kid", input_shape=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.kid_tracker = keras.metrics.Mean()
        self.encoder = keras.Sequential(
            [
                layers.InputLayer(input_shape=input_shape),
                preprocessing.Rescaling(255),
                preprocessing.Resizing(height=75, width=75),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False, input_shape=(75, 75, 3), weights="imagenet"
                ),
                layers.GlobalAveragePooling2D(),
            ]
        )

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        features_dimensions = tf.cast(tf.shape(real_features)[1], dtype=tf.float32)

        kernel_real = (
            real_features @ tf.transpose(real_features) / features_dimensions + 1
        ) ** 3
        kernel_generated = (
            generated_features @ tf.transpose(generated_features) / features_dimensions
            + 1
        ) ** 3
        kernel_cross = (
            real_features @ tf.transpose(generated_features) / features_dimensions + 1
        ) ** 3

        kid = (
            tf.reduce_sum(kernel_real * (1 - tf.eye(batch_size)))
            / (batch_size_f * (batch_size_f - 1))
            + tf.reduce_sum(kernel_generated * (1 - tf.eye(batch_size)))
            / (batch_size_f * (batch_size_f - 1))
            - 2 * tf.reduce_mean(kernel_cross)
        )

        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_states(self):
        self.kid_tracker.reset_states()


# class FID(keras.metrics.Metric):
#     def __init__(self, name='fid', **kwargs):
#         super().__init__(name=name, **kwargs)

#         self.encoder = keras.Sequential([
#             layers.InputLayer(input_shape=(32, 32, 3)),
#             preprocessing.Rescaling(255),
#             preprocessing.Resizing(height=75, width=75),
#             layers.Lambda(keras.applications.inception_v3.preprocess_input),
#             keras.applications.InceptionV3(include_top=False, input_shape=(75, 75, 3), weights="imagenet"),
#             layers.GlobalAveragePooling2D(),
#         ])

#         self.bank_size = 5794 // batch_size * batch_size
#         self.feature_dimensions = self.encoder.output_shape[1]

#         self.real_feature_bank = tf.Variable(
#             tf.zeros(shape=(self.bank_size, self.feature_dimensions)),
#             trainable=False,
#         )
#         self.generated_feature_bank = tf.Variable(
#             tf.zeros(shape=(self.bank_size, self.feature_dimensions)),
#             trainable=False,
#         )
#         self.bank_index = tf.Variable(0)

#     def update_state(self, real_images, generated_images, sample_weight=None):
#         real_features = self.encoder(real_images, training=False)
#         generated_features = self.encoder(generated_images, training=False)

#         self.real_feature_bank[self.bank_index : self.bank_index + batch_size].assign(real_features)
#         self.generated_feature_bank[self.bank_index : self.bank_index + batch_size].assign(generated_features)
#         self.bank_index.assign_add(batch_size)

#     def result(self):
#         if self.bank_index < self.bank_size:
#             return 0.0
#         else:
#             real_mean = tf.math.reduce_mean(self.real_feature_bank, axis=0)
#             generated_mean = tf.math.reduce_mean(self.generated_feature_bank, axis=0)

#             real_covariance = tf.transpose(self.real_feature_bank) @ self.real_feature_bank / self.bank_size - tf.reduce_sum(real_mean ** 2)
#             generated_covariance = tf.transpose(self.generated_feature_bank) @ self.generated_feature_bank / self.bank_size - tf.reduce_sum(generated_mean ** 2)

#             tf.print(tf.shape(real_covariance), tf.shape(generated_covariance))

#             fid = tf.reduce_sum((real_mean - generated_mean) ** 2) + tf.linalg.trace(tf.linalg.sqrtm(real_covariance @ tf.transpose(generated_covariance)))
#             return fid

#     def reset_state(self):
#         self.real_feature_bank.assign(tf.zeros(shape=(self.bank_size, self.feature_dimensions)))
#         self.generated_feature_bank.assign(tf.zeros(shape=(self.bank_size, self.feature_dimensions)))
#         self.bank_index.assign(0)