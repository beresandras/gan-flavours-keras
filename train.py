import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras

from dataset import prepare_dataset
from architecture import get_generator, get_discriminator
from algorithms import MiniMaxGAN, NonSaturatingGAN, LeastSquaresGAN, HingeGAN

tf.get_logger().setLevel("WARN")  # suppress info-level logs

# hyperparameters

# data
num_epochs = 300
image_size = 64
padding = 0.25

# optimization
batch_size = 128
one_sided_label_smoothing = 0.0  # 0.1
ema = 0.99
generator_lr = 2e-4
discriminator_lr = 2e-4
beta_1 = 0.5
beta_2 = 0.999

# architecture (g: generator, d: discriminator)
noise_size = 64
width = 128
initializer = "glorot_uniform"
residual = False
g_transposed = True  # transposed convs vs upsampling + convs in g
g_interpolation = "bilinear"  # only used when the g is residual or not transposed
d_leaky_relu_slope = 0.2
d_dropout_rate = 0.4
d_spectral_norm = False

# adaptive discriminator augmentation
target_accuracy = None  # 0.8  # set to None to disable
integration_steps = 1000

offset_id = 0
id = 2

# load dataset
train_dataset = prepare_dataset("train", image_size, padding, batch_size)
test_dataset = prepare_dataset("test", image_size, padding, batch_size)

# create model
model = NonSaturatingGAN(
    id=offset_id + id,
    generator=get_generator(
        noise_size, width, initializer, residual, g_transposed, g_interpolation
    ),
    discriminator=get_discriminator(
        image_size,
        width,
        initializer,
        residual,
        d_leaky_relu_slope,
        d_dropout_rate,
        d_spectral_norm,
    ),
    one_sided_label_smoothing=one_sided_label_smoothing,
    ema=ema,
    target_accuracy=target_accuracy,
    integration_steps=integration_steps,
)

# optimizers
model.compile(
    generator_optimizer=keras.optimizers.Adam(
        learning_rate=generator_lr, beta_1=beta_1, beta_2=beta_2
    ),
    discriminator_optimizer=keras.optimizers.Adam(
        learning_rate=discriminator_lr, beta_1=beta_1, beta_2=beta_2
    ),
)

# checkpointing
checkpoint_path = "checkpoints/model_{}".format(offset_id + id)
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
plt.savefig(
    "images/_{}_final_{:.3f}.png".format(
        offset_id + id, min(history.history["val_kid"])
    )
)
plt.close()

# save history
plt.figure(figsize=(6, 4))
plt.plot(history.history["val_kid"])
plt.xlabel("epochs")
plt.ylabel("KID")
plt.yscale("log")
plt.tight_layout()
try:
    plt.savefig("graphs/kid_{}.png".format(offset_id + id))
except FileNotFoundError:
    pass
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history.history["aug_p"])
plt.xlabel("epochs")
plt.ylabel("augmentation probability")
plt.tight_layout()
try:
    plt.savefig("graphs/aug_p_{}.png".format(offset_id + id))
except FileNotFoundError:
    pass
plt.close()