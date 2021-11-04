import tensorflow as tf

from tensorflow import keras
from model import GAN


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
        # note: in the paper, real_labels = 1.0, generated_labels = 0.0 is used
        # I changed that to real_labels = 1.0, generated_labels = -1.0
        # to make 0.0 the decision boundary, similarly to the other losses
        # this should make no difference when the discriminator is unregularized

        batch_size = tf.shape(real_logits)[0]
        real_labels = tf.ones(shape=(batch_size, 1))
        generated_labels = -tf.ones(shape=(batch_size, 1))

        generator_loss = keras.losses.mean_squared_error(real_labels, generated_logits)
        discriminator_loss = keras.losses.mean_squared_error(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)


class WassersteinGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        # note: theoretically, if the discriminator is not Lipschitz-constrained,
        # these loss terms can grow indefinitely

        generator_loss = -generated_logits
        discriminator_loss = -real_logits + generated_logits

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)


class HingeGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        generator_loss = -generated_logits
        discriminator_loss = -tf.minimum(1.0, real_logits) + tf.maximum(
            -1.0, generated_logits
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)


class RelativisticGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        batch_size = tf.shape(real_logits)[0]
        real_labels = tf.ones(shape=(batch_size, 1))

        generator_loss = keras.losses.binary_crossentropy(
            real_labels, generated_logits - real_logits, from_logits=True
        )
        discriminator_loss = keras.losses.binary_crossentropy(
            real_labels, real_logits - generated_logits, from_logits=True
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)


class RelativisticAverageGAN(GAN):
    def adversarial_loss(self, real_logits, generated_logits):
        batch_size = tf.shape(real_logits)[0]
        real_labels = tf.ones(shape=(batch_size, 1))

        generator_loss = keras.losses.binary_crossentropy(
            real_labels,
            generated_logits - tf.reduce_mean(real_logits),
            from_logits=True,
        )
        discriminator_loss = keras.losses.binary_crossentropy(
            real_labels,
            real_logits - tf.reduce_mean(generated_logits),
            from_logits=True,
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)