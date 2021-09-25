import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from dataset import prepare_dataset
from algorithms import NonSaturatingGAN

tf.get_logger().setLevel("WARN")  # suppress info-level logs

# hyperparameters
num_epochs = 400
image_size = 32
image_channels = 3
padding = 0.25
batch_size = 128
noise_size = 64
width = 128
one_sided_label_smoothing = 0.1
leaky_relu_slope = 0.2
dropout_rate = 0.4
initializer = keras.initializers.RandomNormal(stddev=0.02)
ema = 0.99

# load STL10 dataset
train_dataset = prepare_dataset("train", image_size, padding, batch_size)
test_dataset = prepare_dataset("test", image_size, padding, batch_size)

# select an algorithm
for Algorithm in [NonSaturatingGAN]:

    # architecture
    model = Algorithm(
        generator=keras.Sequential(
            [
                layers.InputLayer(input_shape=(1, 1, noise_size)),
                layers.Conv2DTranspose(
                    width,
                    kernel_size=4,
                    kernel_initializer=initializer,
                    use_bias=False,
                ),
                layers.BatchNormalization(scale=False),
                layers.ReLU(),
                layers.Reshape(target_shape=(4, 4, width)),
                layers.Conv2DTranspose(
                    width,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=False,
                ),
                layers.BatchNormalization(scale=False),
                layers.ReLU(),
                layers.Conv2DTranspose(
                    width,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=False,
                ),
                layers.BatchNormalization(scale=False),
                layers.ReLU(),
                layers.Conv2DTranspose(
                    image_channels,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    activation="sigmoid",
                ),
            ],
            name="generator",
        ),
        discriminator=keras.Sequential(
            [
                layers.InputLayer(input_shape=(image_size, image_size, image_channels)),
                layers.Conv2D(
                    width,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=False,
                ),
                layers.BatchNormalization(scale=False),
                layers.LeakyReLU(alpha=leaky_relu_slope),
                layers.Conv2D(
                    width,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=False,
                ),
                layers.BatchNormalization(scale=False),
                layers.LeakyReLU(alpha=leaky_relu_slope),
                layers.Conv2D(
                    width,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=False,
                ),
                layers.BatchNormalization(scale=False),
                layers.LeakyReLU(alpha=leaky_relu_slope),
                layers.Dropout(dropout_rate),
                layers.Conv2D(
                    1,
                    kernel_size=4,
                    kernel_initializer=initializer,
                ),
                layers.Flatten(),
            ],
            name="discriminator",
        ),
        one_sided_label_smoothing=one_sided_label_smoothing,
        ema=ema,
    )

    # optimizers
    model.compile(
        generator_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        discriminator_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    )

    # run training
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
        callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)],
    )
