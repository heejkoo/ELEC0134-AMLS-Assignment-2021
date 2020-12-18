import tensorflow as tf


def build_convnet(in_height, in_width, num_classes, gap, init):
    """Build Custom CNN"""

    # Set layer initialization. Default is set to Glorot Normal Initialization.
    if init == 'xavier':
        kernel_init = 'glorot_normal'
    elif init == 'random':
        kernel_init = 'random_normal'
    elif init == 'he':
        kernel_init = 'he_normal'
    else:
        raise NotImplementedError

    # Use Global Pooling or Flatten
    if gap is True:
        flatten = 'globalavgpooling'
    elif gap is False:
        flatten = 'flatten'

    # Build a model, stacking linear layers
    convnet = tf.keras.models.Sequential(name="CustomCNN_using_{}_and_{}_init".format(flatten, init))

    # Add continuously
    convnet.add(tf.keras.layers.Conv2D(filters=32,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='valid',
                                       activation='relu',
                                       input_shape=(in_height, in_width, 3),
                                       kernel_initializer=kernel_init))
    convnet.add(tf.keras.layers.BatchNormalization())
    convnet.add(tf.keras.layers.MaxPooling2D((2, 2)))

    convnet.add(tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='valid',
                                       activation='relu',
                                       kernel_initializer=kernel_init))
    convnet.add(tf.keras.layers.BatchNormalization())
    convnet.add(tf.keras.layers.MaxPooling2D((2, 2)))

    convnet.add(tf.keras.layers.Conv2D(filters=128,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='valid',
                                       activation='relu',
                                       kernel_initializer=kernel_init))
    convnet.add(tf.keras.layers.BatchNormalization())

    # Use Global average pooling or flatten
    if gap is True:
        convnet.add(tf.keras.layers.GlobalAveragePooling2D())
    elif gap is False:
        convnet.add(tf.keras.layers.Flatten())

    # Add classifier part
    convnet.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer=kernel_init))
    convnet.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return convnet


def build_pretrained_vgg(in_height, in_width, num_classes, gap, freeze):
    """Build Pretrained VGGNet"""

    # Get Backbone
    backbone = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(in_height, in_width, 3))

    # Add Sequentially
    x = backbone.output

    # Use Global average pooling or flatten
    if gap is True:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif gap is False:
        x = tf.keras.layers.Flatten()(x)

    # Add classifier
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Use Global Pooling or Flatten
    if gap is True:
        flatten = 'globalavgpooling'
    elif gap is False:
        flatten = 'flatten'

    if freeze is True:
        freezing = 'freezed'
    elif freeze is False:
        freezing = 'notFreezed'

    model = tf.keras.Model(inputs=backbone.input, outputs=predictions, name="{}_VGG16_using_{}".format(freezing, flatten))

    # Freeze except for classifier part
    if freeze is True:
        for layer in backbone.layers:
            layer.trainable = False

    return model