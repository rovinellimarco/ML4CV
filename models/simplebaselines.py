import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import layers


def simple_baselines_heatmap_head(backbone_out, transpose_filters, cfg):
    normal_initializer = initializers.TruncatedNormal(0, 0.01)
    msra_initializer = initializers.VarianceScaling()
    xavier_initializer = initializers.GlorotUniform()

    out = layers.Conv2DTranspose(filters=transpose_filters, kernel_size=4, strides=2, padding='same', activation='relu',
                                 kernel_initializer=normal_initializer,
                                 trainable=True)(backbone_out, training=False)  # for batch normalization
    out = layers.BatchNormalization(momentum=0.1)(out)
    out = layers.Conv2DTranspose(filters=transpose_filters, kernel_size=4, strides=2, padding='same', activation='relu',
                                 kernel_initializer=normal_initializer,
                                 trainable=True)(out)
    out = layers.BatchNormalization(momentum=0.1)(out)
    out = layers.Conv2DTranspose(filters=transpose_filters, kernel_size=4, strides=2, padding='same', activation='relu',
                                 kernel_initializer=normal_initializer,
                                 trainable=True)(out)
    out = layers.BatchNormalization(momentum=0.1)(out)
    out = layers.Conv2D(filters=cfg['num_kps'], kernel_size=1, strides=1, padding='same', activation='linear',
                        kernel_initializer=msra_initializer,
                        trainable=True)(out)

    return out


def create_simplebaselines_model(backbone_fn, transpose_filters, cfg):
    backbone = backbone_fn(include_top=False, weights='imagenet', input_shape=(*cfg['input_shape'], 3))
    backbone.trainable = False
    # backbone.summary()
    block_fmps = []
    last_layer_index = 1
    heatmaps = simple_baselines_heatmap_head(backbone.output, transpose_filters, cfg)
    return tf.keras.Model(inputs=backbone.inputs, outputs=heatmaps)