from tensorflow.keras import layers
import tensorflow as tf


def ResidualBlock(inputs, filters):
    identity = inputs

    x = layers.BatchNormalization(momentum=0.9)(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=filters // 2,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=filters // 2,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)

    x = layers.Add()([identity, x])
    return x


def HourglassConnection(low_res_input, high_res_input, filters):
    # l == lr_in, h == hr_in
    l = low_res_input
    h = high_res_input
    l = ResidualBlock(l, filters)
    l = layers.UpSampling2D()(l)

    h = ResidualBlock(h, filters)
    h = ResidualBlock(h, filters)

    res = layers.Add()([l, h])
    return res


def StackedHourglassModel(backbone_fn, cfg, feature_size=256):
    backbone = backbone_fn(include_top=False, weights='imagenet', input_shape=(*cfg['input_shape'], 3))
    backbone.trainable = False
    # backbone.summary()
    if backbone_fn == tf.keras.applications.MobileNetV3Large:
        layer_indices = [275, 198, 91, 34]
        print('Using MobileNetV3Large')
        feature_maps = [layers.Conv2D(feature_size, 1)(backbone.layers[idx].output) for idx in layer_indices]
    elif backbone_fn == tf.keras.applications.MobileNetV3Small:
        print('Using MobileNetV3Small')
        layer_indices = [242, 165, 46, 25]
        feature_maps = [layers.Conv2D(feature_size, 1)(backbone.layers[idx].output) for idx in layer_indices]
    else:
        print('Using EfficientNet')
        layer_names = ['top_activation', 'block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation']
        feature_maps = [layers.Conv2D(feature_size, 1)(backbone.get_layer(layer_name).output) for layer_name in
                        layer_names]

    result = feature_maps[0]
    result = ResidualBlock(result, feature_size)
    # add another low res layer?
    for i in range(1, len(feature_maps)):
        hr_fmap = feature_maps[i]
        result = HourglassConnection(result, hr_fmap, feature_size)

    heatmaps = layers.Conv2D(filters=cfg['num_kps'], kernel_size=1, strides=1, activation='linear')(result)
    return tf.keras.Model(inputs=backbone.inputs, outputs=heatmaps)
