"""Create the input data pipeline using `tf.data`"""
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from png format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image, label

def random_crop(img):
    scales = list(np.arange(0.8, 1.0, 0.7))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]
    # Create different crops for an image
    crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(64, 64))
    # Return a random crop
    return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
    
def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
       
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    
    image = tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta= 32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    #image = tf.contrib.image.rotate(image, random.uniform(-15, 15), interpolation='NEAREST', name=None)
    
    # Crop image with 50% chance
    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    image = tf.cond(choice < 0.5, lambda: image, lambda: random_crop(image))
    
    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def input_fn(is_training, filenames, labels, params):
    """Input function for the SuperHero dataset.

    The filenames have format "{label}_{id}_{count}.png".
    For instance: "data_dir/groot_001_1.png".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
