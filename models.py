import matplotlib.pyplot as plt
import tensorflow as tf

from abc import abstractmethod
from tensorflow import keras

from augmentation import AdaptiveAugmenter
from metrics import KID


class GAN(keras.Model):
    def __init__(
        self,
        id,
        generator,
        discriminator,
        one_sided_label_smoothing,
        ema,
        target_accuracy,
        integration_steps,
    ):
        super().__init__()

        self.id = id

        self.generator = generator
        self.ema_generator = keras.models.clone_model(self.generator)
        self.discriminator = discriminator
        self.augmenter = AdaptiveAugmenter(
            target_accuracy=target_accuracy,
            integration_steps=integration_steps,
            input_shape=self.generator.output_shape[1:],
        )

        self.noise_size = self.generator.input_shape[3]
        self.one_sided_label_smoothing = one_sided_label_smoothing
        self.ema = ema

        self.latent_samples = None

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")
        self.kid = KID(input_shape=self.generator.output_shape[1:])

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
            self.augmentation_probability_tracker,
            self.kid,
        ]

    def generate(self, batch_size, training):
        latent_samples = tf.random.normal(shape=(batch_size, 1, 1, self.noise_size))
        if training:
            generated_images = self.generator(latent_samples, training)
        else:
            generated_images = self.ema_generator(latent_samples, training)
        return generated_images

    def plot_images(self, epoch, logs, num_rows=2, num_cols=8, interval=5):
        if (epoch + 1) % interval == 0:
            num_images = num_rows * num_cols
            if self.latent_samples is None:
                self.latent_samples = tf.random.normal(
                    shape=(num_images, 1, 1, self.noise_size)
                )
            generated_images_1 = self.ema_generator(self.latent_samples, training=False)
            generated_images_2 = self.generate(num_images, training=False)

            plt.figure(figsize=(num_cols * 1.5, 2 * num_rows * 1.5))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(2 * num_rows, num_cols, index + 1)
                    plt.imshow(generated_images_1[index])
                    plt.axis("off")
                    plt.subplot(2 * num_rows, num_cols, num_images + index + 1)
                    plt.imshow(generated_images_2[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                "images/{}_{}_{:.3f}.png".format(self.id, epoch + 1, self.kid.result())
            )
            plt.close()

    @abstractmethod
    def adversarial_loss(self, real_logits, generated_logits):
        pass

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generate(batch_size, training=True)
            real_logits = self.discriminator(
                self.augmenter(real_images, training=True), training=True
            )
            generated_logits = self.discriminator(
                self.augmenter(generated_images, training=True), training=True
            )
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

        self.augmenter.update(real_logits)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy.update_state(1.0, tf.keras.activations.sigmoid(real_logits))
        self.generated_accuracy.update_state(
            0.0, tf.keras.activations.sigmoid(generated_logits)
        )
        self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        for weight, ema_weight in zip(
            self.generator.weights, self.ema_generator.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        generated_images = self.generate(batch_size, training=False)

        self.kid.update_state(real_images, generated_images)

        return {self.kid.name: self.kid.result()}