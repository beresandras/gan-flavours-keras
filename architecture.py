from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow_addons.layers import SpectralNormalization


def spectral_norm_wrapper(layer, spectral_norm):
    if spectral_norm:
        return SpectralNormalization(layer)
    else:
        return layer


def get_generator(noise_size, width, initializer, residual, transposed):
    input = layers.Input(shape=(1, 1, noise_size))

    x = layers.Conv2DTranspose(
        width,
        kernel_size=4,
        kernel_initializer=initializer,
        use_bias=False,
    )(input)

    for _ in range(3):
        if residual:
            x_skip = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)
        if transposed:
            x = layers.Conv2DTranspose(
                width,
                kernel_size=4,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=residual,
            )(x)
        else:
            x = layers.UpSampling2D(size=2, interpolation="nearest")(x)
            x = layers.Conv2D(
                width,
                kernel_size=4,
                padding="same",
                kernel_initializer=initializer,
                use_bias=residual,
            )(x)
        if residual:
            x = layers.Add()([x_skip, x])
            x = preprocessing.Rescaling(scale=0.5 ** 0.5)(x)

    x = layers.BatchNormalization(scale=False)(x)
    x = layers.ReLU()(x)
    if transposed:
        output = layers.Conv2DTranspose(
            3,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            activation="sigmoid",
        )(x)
    else:
        x = layers.UpSampling2D(size=2, interpolation="nearest")(x)
        output = layers.Conv2D(
            3,
            kernel_size=4,
            padding="same",
            kernel_initializer=initializer,
            activation="sigmoid",
        )(x)

    return keras.Model(input, output, name="generator")


def get_discriminator(
    image_size,
    width,
    initializer,
    residual,
    leaky_relu_slope,
    dropout_rate,
    spectral_norm,
):
    input = layers.Input(shape=(image_size, image_size, 3))

    x = spectral_norm_wrapper(
        layers.Conv2D(
            width,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        ),
        spectral_norm=spectral_norm,
    )(input)

    for _ in range(3):
        if residual:
            x_skip = layers.AveragePooling2D(pool_size=2)(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)
        x = spectral_norm_wrapper(
            layers.Conv2D(
                width,
                kernel_size=4,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=residual,
            ),
            spectral_norm=spectral_norm,
        )(x)
        if residual:
            x = layers.Add()([x_skip, x])
            x = preprocessing.Rescaling(scale=0.5 ** 0.5)(x)

    x = layers.BatchNormalization(scale=False)(x)
    x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = spectral_norm_wrapper(
        layers.Conv2D(1, kernel_size=4, kernel_initializer=initializer),
        spectral_norm=spectral_norm,
    )(x)
    output = layers.Flatten()(x)

    return keras.Model(input, output, name="discriminator")
