import matplotlib.pyplot as plt

import tensorflow as tf


# "hard sigmoid", useful for binary accuracy calculation from logits
def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


def generate_images_with(model, history, id, num_rows=8, num_cols=8, is_jupyter=False):
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
    if is_jupyter:
        plt.show()
    else:
        plt.savefig(
            "images/_{}_final_{:.3f}.png".format(id, min(history.history["val_kid"]))
        )
    plt.close()


def plot_history(history, id, is_jupyter=False):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["val_kid"])
    plt.xlabel("epochs")
    plt.ylabel("KID")
    plt.yscale("log")
    plt.tight_layout()
    if is_jupyter:
        plt.show()
    else:
        plt.savefig("graphs/kid_{}.png".format(id))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["aug_p"])
    plt.xlabel("epochs")
    plt.ylabel("augmentation probability")
    plt.tight_layout()
    if is_jupyter:
        plt.show()
    else:
        plt.savefig("graphs/aug_p_{}.png".format(id))
    plt.close()
