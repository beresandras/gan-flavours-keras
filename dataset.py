import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_image_factory(image_size, padding):
    def preprocess_image(data):
        height = tf.cast(tf.shape(data["image"])[0] - 1, dtype=tf.float32)
        width = tf.cast(tf.shape(data["image"])[1] - 1, dtype=tf.float32)
        bounding_box = data["bbox"] * tf.stack([height, width, height, width])

        target_center_y = (bounding_box[0] + bounding_box[2]) / 2
        target_center_x = (bounding_box[1] + bounding_box[3]) / 2
        target_size = tf.maximum(
            (1 + padding) * (bounding_box[2] - bounding_box[0]),
            (1 + padding) * (bounding_box[3] - bounding_box[1]),
        )
        target_height = tf.reduce_min(
            [target_size, 2 * target_center_y, 2 * (height - target_center_y)]
        )
        target_width = tf.reduce_min(
            [target_size, 2 * target_center_x, 2 * (width - target_center_x)]
        )

        image = tf.image.crop_to_bounding_box(
            data["image"],
            offset_height=tf.cast(
                tf.math.rint(target_center_y - target_height / 2), dtype=tf.int32
            ),
            offset_width=tf.cast(
                tf.math.rint(target_center_x - target_width / 2), dtype=tf.int32
            ),
            target_height=tf.cast(tf.math.rint(target_height), dtype=tf.int32) + 1,
            target_width=tf.cast(tf.math.rint(target_width), dtype=tf.int32) + 1,
        )
        image = tf.image.resize(
            image, size=[image_size, image_size], method=tf.image.ResizeMethod.AREA
        )
        return tf.clip_by_value(image / 255, 0, 1)

    return preprocess_image


def prepare_dataset(split, image_size, padding, batch_size):
    return (
        tfds.load(
            "caltech_birds2011",
            data_dir="D:/Documents/datasets/tensorflow/",
            split=split,
            shuffle_files=True,
        )
        .map(
            preprocess_image_factory(image_size, padding),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .cache()
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )