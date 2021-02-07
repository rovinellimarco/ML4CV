from functools import partial

import tensorflow as tf
import numpy as np
from datasets.posetrack import posetrack_annotations_generator
from datasets.coco import annotations_generator


def to_float(d):
    return tf.cast(d, tf.float32)


def read_images(img_id,  # int32
                image_filepaths,  # string
                bbox,  # (n, 4) - float32 (x,y,w,h)
                annotations  # (n, K, 3) - int32
                ):
    img = tf.io.read_file(image_filepaths)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.expand_dims(img, axis=0)
    return img_id, img, bbox, annotations


def read_images_batched(img_id,  # int32
                        image_filepaths,  # string
                        bbox,  # (n, 4) - float32 (x,y,w,h)
                        annotations  # (n, K, 3) - int32
                        ):
    # if len(tf.shape(image_filepaths)) == 0:
    image_filepaths = tf.reshape(image_filepaths, (-1,))
    # image_filepaths = tf.unstack(image_filepaths)
    # print('images',len(image_filepaths))
    images = []

    for image_filepath in image_filepaths:
        img = tf.io.read_file(image_filepath)
        img = tf.io.decode_jpeg(img, channels=3)
        images.append(img)
    img = tf.stack(images)
    # print('images read', img.shape)
    # else:
    #     img = tf.io.read_file(image_filepaths)
    #     img = tf.io.decode_jpeg(img, channels=3)
    return img_id, img, bbox, annotations


def input_data_from_annotations(img_id,  # n, int32
                                img,  # n, height, width, 3
                                bboxes,  # n, 4 - float32 (x,y,w,h)
                                annotations,  # n, K, 3 - int32 (x,y,valid)
                                cfg,
                                stage='train'):
    if not isinstance(img_id, tf.Tensor):
        # raise ValueError('Using a non tensor input to the mapping function')
        img_id = tf.constant(img_id)
        img = tf.constant(img)
        bboxes = tf.constant(bboxes)
        annotations = tf.constant(annotations)
        raise ValueError('Needs tensors')

    model_input_shape = cfg['input_shape']
    model_aspect_ratio = cfg['input_shape'][1] / cfg['input_shape'][0]

    # img = tf.expand_dims(img, axis=0) if len(tf.shape(img)) == 3 else img
    # print(len(tf.shape(img)))
    image_wh = to_float(tf.expand_dims(tf.shape(img)[1:3], axis=0))

    n = tf.shape(bboxes)[0]
    k = tf.shape(annotations)[1]

    xy_bl = bboxes[:, :2]  # n, 2
    wh = bboxes[:, 2:]  # n, 2
    xy_tr = xy_bl + wh  # n, 2
    # bbox sanitization
    center = xy_bl + wh * 0.5  # n, 2
    wh = tf.clip_by_value(wh,
                          wh[:, ::-1] * [model_aspect_ratio, 1 / model_aspect_ratio],
                          to_float(tf.shape(img)[1:3]))
    wh = wh  # * (1+cfg['margin_factor'])
    xy_bl = center - wh * 0.5
    xy_tr = center + wh * 0.5

    # if h < w / aspect_ratio:
    #     h = w / aspect_ratio
    # elif w < aspect_ratio * h:
    #     w = h * aspect_ratio

    # TODO margin handling
    crops = tf.concat([xy_bl[:, ::-1], xy_tr[:, ::-1]], axis=1)  # n, 4
    hw = wh[:, ::-1]

    # plt.imshow(img.numpy()[0,:,:,:].astype(np.int))
    # print(crops.numpy())
    # plt.scatter([crops[0,1].numpy(), crops[0,3].numpy()],
    #             [crops[0,0].numpy(), crops[0,2].numpy()], c=['r', 'c'])
    # plt.show()

    crops = crops / tf.concat([image_wh, image_wh], axis=1)  # n, 4
    # this could be done using a batched input
    # n, model_input_h, model_input_w, 3
    box_indices = tf.zeros((n,), dtype=np.int32) if cfg['coco_dataset'] else tf.range(tf.shape(img_id)[0])
    # print(img.shape, crops.shape, box_indices.shape)
    input_crops = tf.image.crop_and_resize(img,
                                           crops,
                                           box_indices,
                                           crop_size=model_input_shape,
                                           extrapolation_value=0)

    # plt.imshow(input_crops[0,:,:,:].numpy().astype(int))
    # plt.show()
    # input()

    # keypoint correction
    annotations = to_float(annotations)
    joints_coord = (annotations[:, :, :2] - tf.reshape(xy_bl, (n, 1, 2))) / tf.reshape(hw, (n, 1, 2))  # n, k, 2
    joints_valid = annotations[:, :, -1]  # n, 1
    if stage == 'train':
        return input_crops, joints_coord, joints_valid, crops
    else:
        crops *= tf.concat([image_wh, image_wh], axis=1)
        # print(len(tf.shape(img_id)), img_id.shape)
        images_ids = tf.repeat(img_id, n) if len(tf.shape(img_id)) == 0 else img_id
        return input_crops, images_ids, crops


def target_from_joints(input_crops,  # bs, h, w, 3
                       joints_coord,  # bs, k, 2   # NORMALIZED
                       joints_valid,  # bs, k
                       crops,  # bs, 4 - not relevant here
                       cfg,
                       output_relative):
    # possibly this is batched
    numpy_input = False
    if not isinstance(joints_coord, tf.Tensor):
        joints_coord = tf.expand_dims(tf.constant(joints_coord), axis=0)
        numpy_input = True

    model_input_shape = cfg['input_shape']
    model_output_shape = cfg['output_shape'] if output_relative == True else output_relative
    sigma = to_float(cfg['sigma'])
    k = cfg['num_kps']
    bs = tf.shape(joints_coord)[0]

    joints_coord *= model_output_shape

    x = tf.range(model_output_shape[1])
    y = tf.range(model_output_shape[0])
    xx, yy = tf.meshgrid(x, y)  # bs, w, h (each one)
    xx = to_float(tf.reshape(xx, (1, *model_output_shape, 1)))  # 1, h, w, 1
    yy = to_float(tf.reshape(yy, (1, *model_output_shape, 1)))  # 1, h, w, 1
    # bs, 1, 1, k
    x = tf.reshape(joints_coord[:, :, 0], (bs, 1, 1, k))
    x = to_float(tf.floor(0.5 + x))
    # bs, 1, 1, k
    y = tf.reshape(joints_coord[:, :, 1], [bs, 1, 1, k])
    y = to_float(tf.floor(y + 0.5))

    heatmap = tf.exp(-(((xx - x) / sigma) ** 2) / to_float(2.0) - \
                     (((yy - y) / sigma) ** 2) / to_float(2.0))  # bs, out_w, out_h, k

    if numpy_input:
        return heatmap.numpy().squeeze()
    else:
        return input_crops, heatmap, joints_valid


def create_dataset(cfg, coco, data_type, stage='train'):
    if coco is not None:
        gen_func = partial(annotations_generator, cfg=cfg, coco=coco, data_type=data_type)
    else:
        gen_func = partial(posetrack_annotations_generator, cfg=cfg, data_type=data_type, only_existing_folders=True)
    dataset = tf.data.Dataset.from_generator(gen_func,
                                             output_types=(tf.int32, tf.string, tf.float32, tf.int32))
    map_func = partial(input_data_from_annotations, cfg=cfg, stage=stage)
    heat_func = partial(target_from_joints, cfg=cfg, output_relative=True)

    if coco is None:
        py_read_images = lambda *x: tf.py_function(func=read_images_batched, inp=x,
                                                   Tout=(tf.int32, tf.uint8, tf.float32, tf.int32))
    else:
        py_read_images = read_images
    img_dat = dataset.map(py_read_images)
    map_dat = img_dat.map(map_func)

    if coco is not None:
        map_dat = map_dat.unbatch().shuffle(256).batch(cfg['batch_size'])
    else:
        pass  # shuffling

    if stage == 'train':
        heat_dat = map_dat.map(heat_func)
    else:
        heat_dat = map_dat
    prefetched_dataset = heat_dat.prefetch(4)
    return prefetched_dataset
