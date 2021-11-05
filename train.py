import os
import matplotlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras

from dataset import prepare_dataset
from architecture import get_generator, get_discriminator
from losses import (
    MiniMaxGAN,
    NonSaturatingGAN,
    LeastSquaresGAN,
    HingeGAN,
    WassersteinGAN,
    RelativisticGAN,
    RelativisticAverageGAN,
)
from utils import generate_images_with, plot_history

tf.get_logger().setLevel("WARN")  # suppress info-level logs

matplotlib.use("Agg")

# hyperparameters

# data
# some datasets might be unavailable for download at times
dataset_name = "caltech_birds2011"  # "oxford_flowers102", "celeb_a", "cifar10"
image_size = 64  # 64, 64, 32
num_epochs = 400  # 500, 25, 100
plot_interval = 10  # 10, 1, 2

# optimization
batch_size = 128
one_sided_label_smoothing = 0.0  # can be 0.1
ema = 0.99
generator_lr = 2e-4
discriminator_lr = 2e-4
beta_1 = 0.5
beta_2 = 0.999

# architecture
noise_size = 64
depth = 4  # number of up- and downsampling layers, change with resolution
width = 128
initializer = "glorot_uniform"
residual = False
transposed = True  # transposed convs vs upsampling + convs in generator
leaky_relu_slope = 0.2
dropout_rate = 0.4
spectral_norm = False

# adaptive discriminator augmentation
target_accuracy = None  # 0.85, set to None to disable
integration_steps = 1000

id = 0

# load dataset
train_dataset = prepare_dataset(dataset_name, "train", image_size, batch_size)
val_dataset = prepare_dataset(dataset_name, "validation", image_size, batch_size)

# create model
model = NonSaturatingGAN(
    id=id,
    generator=get_generator(
        noise_size, depth, width, initializer, residual, transposed
    ),
    discriminator=get_discriminator(
        image_size,
        depth,
        width,
        initializer,
        residual,
        leaky_relu_slope,
        dropout_rate,
        spectral_norm,
    ),
    one_sided_label_smoothing=one_sided_label_smoothing,
    ema=ema,
    target_accuracy=target_accuracy,
    integration_steps=integration_steps,
    plot_interval=plot_interval,
)

model.compile(
    generator_optimizer=keras.optimizers.Adam(
        learning_rate=generator_lr, beta_1=beta_1, beta_2=beta_2
    ),
    discriminator_optimizer=keras.optimizers.Adam(
        learning_rate=discriminator_lr, beta_1=beta_1, beta_2=beta_2
    ),
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
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
)

# load best model
model.load_weights(checkpoint_path)
generate_images_with(model, history, id)

# plot history
plot_history(history, id)