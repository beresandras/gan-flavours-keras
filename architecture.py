from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow_addons.layers import SpectralNormalization


def get_generator(noise_size, width, initializer, residual, transposed=True):
    input = layers.Input(shape=(1, 1, noise_size))

    x = layers.Conv2DTranspose(
        width,
        kernel_size=4,
        kernel_initializer=initializer,
        use_bias=False,
    )(input)
    x = layers.BatchNormalization(scale=False)(x)
    x = layers.ReLU()(x)

    for _ in range(3):
        x_skip = layers.UpSampling2D(size=2)(x)
        # x_skip = layers.Conv2DTranspose(
        #     width,
        #     kernel_size=2,
        #     strides=2,
        #     padding="same",
        #     kernel_initializer=initializer,
        #     use_bias=False,
        # )(x)
        if transposed:
            x = layers.Conv2DTranspose(
                width,
                kernel_size=4,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            )(x)
        else:
            x = layers.Conv2D(
                width,
                kernel_size=4,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            )(x_skip)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)
        if residual:
            x = layers.Add()([x_skip, x])

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
        x = layers.UpSampling2D(size=2)(x)
        output = layers.Conv2D(
            3,
            kernel_size=4,
            padding="same",
            kernel_initializer=initializer,
            activation="sigmoid",
        )(x)

    return keras.Model(input, output, name="generator")


def get_discriminator(
    image_size, width, initializer, leaky_relu_slope, dropout_rate, residual
):
    input = layers.Input(shape=(image_size, image_size, 3))

    x = layers.Conv2D(
        width,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False,
    )(input)
    x = layers.BatchNormalization(scale=False)(x)
    x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)

    for _ in range(3):
        x_skip = layers.AveragePooling2D(pool_size=2)(x)
        x = layers.Conv2D(
            width,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)
        if residual:
            x = layers.Add()([x_skip, x])

    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(1, kernel_size=4, kernel_initializer=initializer)(x)
    output = layers.Flatten()(x)

    return keras.Model(input, output, name="discriminator")
