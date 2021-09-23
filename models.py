import tensorflow as tf

from abc import abstractmethod
from tensorflow import keras

# from metrics import KID


class GAN(keras.Model):
    def __init__(self, generator, discriminator, one_sided_label_smoothing):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.one_sided_label_smoothing = one_sided_label_smoothing
        self.noise_size = self.generator.input_shape[1]

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        # self.kid = KID()

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
        ]
        # self.kid,

    def generate(self, batch_size, training):
        latent_samples = tf.random.normal(shape=(batch_size, self.noise_size))
        generated_images = self.generator(latent_samples, training)
        return generated_images

    @abstractmethod
    def adversarial_loss(self, real_logits, generated_logits):
        pass

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generate(batch_size, training=True)
            combined_images = tf.concat([real_images, generated_images], axis=0)
            combined_logits = self.discriminator(combined_images, training=True)
            real_logits = combined_logits[:batch_size]
            generated_logits = combined_logits[batch_size:]
            generator_loss, discriminator_loss = self.adversarial_loss(
                real_logits, generated_logits
            )

        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_weights
        )
        discriminator_gradients = tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_weights)
        )

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy(1.0, tf.keras.activations.sigmoid(real_logits))
        self.generated_accuracy(0.0, tf.keras.activations.sigmoid(generated_logits))
        # self.kid.update_state(real_images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        generated_images = self.generate(batch_size, training=False)
        combined_images = tf.concat([real_images, generated_images], axis=0)
        combined_logits = self.discriminator(combined_images, training=False)
        real_logits = combined_logits[:batch_size]
        generated_logits = combined_logits[batch_size:]
        generator_loss, discriminator_loss = self.adversarial_loss(
            real_logits, generated_logits
        )

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy(1.0, tf.keras.activations.sigmoid(real_logits))
        self.generated_accuracy(0.0, tf.keras.activations.sigmoid(generated_logits))
        # self.kid.update_state(real_images, generated_images)

        return {m.name: m.result() for m in self.metrics}