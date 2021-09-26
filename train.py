import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras

from dataset import prepare_dataset
from architecture import get_generator, get_discriminator
from algorithms import NonSaturatingGAN

tf.get_logger().setLevel("WARN")  # suppress info-level logs

# hyperparameters
num_epochs = 160
image_size = 64
padding = 0.25
batch_size = 128
noise_size = 64
width = 128
one_sided_label_smoothing = 0.1
leaky_relu_slope = 0.2
dropout_rate = 0.4
initializer = keras.initializers.RandomNormal(stddev=0.02)
ema = 0.99

residuals = [False]
transposeds = [True]

# load STL10 dataset
train_dataset = prepare_dataset("train", image_size, padding, batch_size)
test_dataset = prepare_dataset("test", image_size, padding, batch_size)

# select an algorithm
for id, (residual, transposed) in enumerate(zip(residuals, transposeds)):

    # architecture
    model = NonSaturatingGAN(
        id=id,
        generator=get_generator(noise_size, width, initializer, residual, transposed),
        discriminator=get_discriminator(
            image_size, width, initializer, leaky_relu_slope, dropout_rate, residual
        ),
        one_sided_label_smoothing=one_sided_label_smoothing,
        ema=ema,
    )

    # optimizers
    model.compile(
        generator_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        discriminator_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    )

    # checkpointing
    checkpoint_path = "checkpoints/model_{}".format(id)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_kid",
        mode="min",
        save_best_only=True,
    )

    # run training
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
        callbacks=[
            keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
            checkpoint_callback,
        ],
    )

    # save history
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["val_kid"][20:])
    plt.xlabel("epochs")
    plt.ylabel("KID")
    plt.tight_layout()
    plt.savefig("graphs/kid_{}.png".format(id))

    model.load_weights(checkpoint_path)

    # generate images
    num_rows = 8
    num_cols = 8
    num_images = num_rows * num_cols
    generated_images = model.generate(num_images, training=False)

    plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            plt.imshow(generated_images[index])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("images/_{}_final.png".format(id))
    plt.close()