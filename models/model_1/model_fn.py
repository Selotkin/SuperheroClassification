"""Define the model."""

import tensorflow as tf
from keras import regularizers
tf.logging.set_verbosity(tf.logging.ERROR)

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
        out = tf.layers.batch_normalization(out, momentum = 0.9)
        return tf.nn.relu(tf.reshape(out, tf.shape(featureMap)), name = scope.name)


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    out = images
    
    conv1 = convLayer(out, 3, 3, 1, 1, 32, "conv1")
    pool1 = maxPoolLayer(conv1, 2, 2, 2, 2, "pool1")
    #dropout1 = dropout(pool1, 0.5)
    
    conv2 = convLayer(pool1, 3, 3, 1, 1, 32, "conv2")
    pool2 = maxPoolLayer(conv2, 2, 2, 2, 2, "pool2")
    dropout1 = dropout(pool2, 0.5)

    conv3 = convLayer(dropout1, 3, 3, 1, 1, 64, "conv3")
    pool3 = maxPoolLayer(conv3, 2, 2, 2, 2, "pool3")
    #dropout3 = dropout(pool3, 0.5)
    
    conv4 = convLayer(pool3, 3, 3, 1, 1, 64, "conv4")
    pool4 = maxPoolLayer(conv4, 2, 2, 2, 2, "pool4")
    #dropout4 = dropout(pool4, 0.5)
    
    conv5 = convLayer(pool4, 3, 3, 1, 1, 128, "conv5")
    pool5 = maxPoolLayer(conv5, 2, 2, 2, 2, "pool5")
    #dropout5 = dropout(pool5, 0.5)
    
    fcIn = tf.reshape(pool5, [-1, 2 * 2 * 128])
    dense1 = tf.layers.dense(fcIn, 128, name = 'dense1', kernel_regularizer=regularizers.l2(0.005))
    dense1r = tf.nn.relu(dense1)
    
    dropout2 = dropout(dense1r, 0.5)
    dense2 = tf.layers.dense(dropout2, params.num_labels, name = 'dense2', kernel_regularizer=regularizers.l2(0.005))
    
    #fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
    #dropout2 = dropout(fc7, 0.5)

    #fc8 = fcLayer(dropout2, 4096, params.num_labels, True, "fc8")
    #print(fc8.shape)
    return dense2


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)
        probabilities = tf.nn.softmax(logits, name="softmax_tensor")

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)

    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    predictions = tf.Print(predictions, [predictions], message="This is predictions: ")
    labels = tf.Print(labels, [labels], message="This is labels: ")
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['probabilities'] = probabilities
    

    if is_training:
        model_spec['train_op'] = train_op
        model_spec['optimizer'] = optimizer

    return model_spec
