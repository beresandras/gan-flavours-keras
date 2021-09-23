import tensorflow as tf

from tensorflow import keras
from models import GAN


class MiniMaxGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        batch_size = tf.shape(real_logits)[0]
        real_labels = tf.ones(shape=(batch_size, 1)) - self.one_sided_label_smoothing
        generated_labels = tf.zeros(shape=(batch_size, 1))

        generator_loss = -keras.losses.binary_crossentropy(
            generated_labels, generated_logits, from_logits=True
        )
        discriminator_loss = keras.losses.binary_crossentropy(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
            from_logits=True,
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)


class NonSaturatingGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        batch_size = tf.shape(real_logits)[0]
        real_labels = tf.ones(shape=(batch_size, 1)) - self.one_sided_label_smoothing
        generated_labels = tf.zeros(shape=(batch_size, 1))

        generator_loss = keras.losses.binary_crossentropy(
            real_labels, generated_logits, from_logits=True
        )
        discriminator_loss = keras.losses.binary_crossentropy(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
            from_logits=True,
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)


class LeastSquaresGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        batch_size = tf.shape(real_logits)[0]
        real_labels = tf.ones(shape=(batch_size, 1))
        generated_labels = tf.zeros(shape=(batch_size, 1))

        generator_loss = keras.losses.mean_squared_error(
            real_labels, generated_logits, from_logits=True
        )
        discriminator_loss = keras.losses.mean_squared_error(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
            from_logits=True,
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)


class WassersteinGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        generator_loss = -generated_logits
        discriminator_loss = generated_logits - real_logits

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)