
import tensorflow as tf
import dataset_characteristics


def normalize_triplets(image_anchor, image_neighbor, image_distant, label_anchor, label_neighbor, label_distant):

    image_anchor = tf.cast(image_anchor, tf.float32) * (1. / 255) - 0.5
    image_neighbor = tf.cast(image_neighbor, tf.float32) * (1. / 255) - 0.5
    image_distant = tf.cast(image_distant, tf.float32) * (1. / 255) - 0.5   
    return image_anchor, image_neighbor, image_distant, label_anchor, label_neighbor, label_distant

def parse_function(serialized):
    IMAGE_SIZE_height = dataset_characteristics.get_image_height()
    IMAGE_SIZE_width = dataset_characteristics.get_image_width()
    IMAGE_PIXELS = IMAGE_SIZE_height * IMAGE_SIZE_width

    features = \
        {
            'image_anchor': tf.io.FixedLenFeature([], tf.string),
            'image_neighbor': tf.io.FixedLenFeature([], tf.string),
            'image_distant': tf.io.FixedLenFeature([], tf.string),
            'label_anchor': tf.io.FixedLenFeature([], tf.int64),
            'label_neighbor': tf.io.FixedLenFeature([], tf.int64),
            'label_distant': tf.io.FixedLenFeature([], tf.int64)
        }

    parsed_example = tf.io.parse_single_example(serialized=serialized,
                                             features=features)

    # image_anchor = tf.decode_raw(parsed_example['image_anchor'], tf.uint8)
    # image_neighbor = tf.decode_raw(parsed_example['image_neighbor'], tf.uint8)
    # image_distant = tf.decode_raw(parsed_example['image_distant'], tf.uint8)

    # https://www.tensorflow.org/api_docs/python
    image_anchor = tf.compat.v1.decode_raw(parsed_example['image_anchor'], tf.uint8)
    image_neighbor = tf.compat.v1.decode_raw(parsed_example['image_neighbor'], tf.uint8)
    image_distant = tf.compat.v1.decode_raw(parsed_example['image_distant'], tf.uint8)
    label_anchor = parsed_example['label_anchor']
    label_neighbor = parsed_example['label_neighbor']
    label_distant = parsed_example['label_distant']

    image_anchor.set_shape((IMAGE_PIXELS))
    image_neighbor.set_shape((IMAGE_PIXELS))
    image_distant.set_shape((IMAGE_PIXELS))

    return image_anchor, image_neighbor, image_distant, label_anchor, label_neighbor, label_distant