import matplotlib.pyplot as plt
import tensorflow as tf

from abc import abstractmethod
from tensorflow import keras

from metrics import KID
from utils import step


class GAN(keras.Model):
    def __init__(
        self,
        id,
        generator,
        discriminator,
        augmenter,
        one_sided_label_smoothing,
        ema,
        kid_image_size,
        plot_interval,
        is_jupyter=False,
    ):
        super().__init__()

        self.id = id

        self.generator = generator
        self.ema_generator = keras.models.clone_model(self.generator)
        self.discriminator = discriminator
        self.augmenter = augmenter

        self.noise_size = self.generator.input_shape[3]
        self.one_sided_label_smoothing = one_sided_label_smoothing
        self.ema = ema
        self.kid_image_size = kid_image_size
        self.plot_interval = plot_interval
        self.is_jupyter = is_jupyter

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
        self.kid = KID(
            input_shape=self.generator.output_shape[1:], image_size=self.kid_image_size
        )

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
        # using ema_generator during inference
        if training:
            generated_images = self.generator(latent_samples, training)
        else:
            generated_images = self.ema_generator(latent_samples, training)
        return generated_images

    def plot_images(self, epoch, logs, num_rows=2, num_cols=8):
        # plot random generated images for visual evaluation of generation quality
        if (epoch + 1) % self.plot_interval == 0:
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
            if self.is_jupyter:
                plt.show()
            else:
                plt.savefig(
                    "images/{}_{}_{:.3f}.png".format(
                        self.id, epoch + 1, self.kid.result()
                    )
                )
            plt.close()

    @abstractmethod
    def adversarial_loss(self, real_logits, generated_logits):
        pass

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # use persistent gradient tape because gradients will be calculated twice
        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generate(batch_size, training=True)

            # gradient is calculated through the image augmentation
            if self.augmenter.target_accuracy is not None:
                real_images = self.augmenter(real_images, training=True)
                generated_images = self.augmenter(generated_images, training=True)

            # separate forward passes for the real and generated images, meaning
            # that batch normalization is applied separately
            real_logits = self.discriminator(real_images, training=True)
            generated_logits = self.discriminator(generated_images, training=True)

            generator_loss, discriminator_loss = self.adversarial_loss(
                real_logits, generated_logits
            )

        # calculate gradients and update weights
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

        # update the augmentation probability based on the discriminator's performance
        if self.augmenter.target_accuracy is not None:
            self.augmenter.update(real_logits)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy.update_state(1.0, step(real_logits))
        self.generated_accuracy.update_state(0.0, step(generated_logits))
        self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        # track the exponential moving average of the generator's weights to decrease
        # variance in the generation quality
        for weight, ema_weight in zip(
            self.generator.weights, self.ema_generator.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        generated_images = self.generate(batch_size, training=False)

        self.kid.update_state(real_images, generated_images)

        # only KID is measured during the evaluation phase for computational efficiency
        return {self.kid.name: self.kid.result()}