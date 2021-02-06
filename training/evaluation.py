import math

import tensorflow as tf
import numpy as np
from tqdm.notebook import tqdm
import time
import os.path as osp
from pycocotools.cocoeval import COCOeval
import json
import pickle


def results_from_model(model, test_dataset, cfg):
    dump_results = []

    start_time = time.time()
    for img_x, image_id, crop_infos in tqdm(test_dataset):
        if isinstance(image_id, tf.Tensor):
            image_id = image_id.numpy()  # .decode('utf-8')
            crop_infos = crop_infos.numpy()

        kps_result = np.zeros((img_x.shape[0], cfg['num_kps'], 3))

        # forward
        heatmap = model(img_x, training=False).numpy()

        for batch_index in range(img_x.shape[0]):
            for j in range(cfg['num_kps']):
                hm_j = heatmap[batch_index, :, :, j]
                y, x = np.unravel_index(hm_j.argmax(), hm_j.shape)

                # refining position
                px = int(math.floor(x + 0.5))
                py = int(math.floor(y + 0.5))
                if 1 < px < cfg['output_shape'][1] - 1 and 1 < py < cfg['output_shape'][0] - 1:
                    diff = np.array([hm_j[py][px + 1] - hm_j[py][px - 1],
                                     hm_j[py + 1][px] - hm_j[py - 1][px]])
                    diff = np.sign(diff)
                    x += diff[0] * .25
                    y += diff[1] * .25
                kps_result[batch_index, j, :2] = (x / cfg['output_shape'][1],
                                                  y / cfg['output_shape'][0])
                kps_result[batch_index, j, 2] = hm_j.max()
                # for k in range(cfg['num_kps']):
                #     y, x = np.unravel_index(y_pred_full[img_idx,:,:,k].argmax(), cfg['output_shape'])
                #     y_coords[k, :] = [x, y]
                # y_coords /= [y_pred.shape[1], y_pred.shape[2]]
                # mapping back to original image
                h_scale = crop_infos[batch_index, 2] - crop_infos[batch_index, 0]
                w_scale = crop_infos[batch_index, 3] - crop_infos[batch_index, 1]
                kps_result[batch_index, j, 0] = kps_result[batch_index, j, 0] * w_scale + crop_infos[batch_index][1]
                kps_result[batch_index, j, 1] = kps_result[batch_index, j, 1] * h_scale + crop_infos[batch_index][0]

            # DEBUG
            # image_coco = coco_val.loadImgs([image_id[batch_index].item()])[0]
            # I = io.imread('./images/val2017/' + image_coco['file_name'])
            # fig, axs = plt.subplots(1,1)
            # axs.imshow(I)
            # axs.scatter(kps_result[batch_index, :, 0], kps_result[batch_index, :, 1])
            # plt.show()
            # time.sleep(5)
            ############################

        score_result = np.copy(kps_result[:, :, 2])
        kps_result[:, :, 2] = 1
        kps_result = kps_result.reshape(-1, cfg['num_kps'] * 3)

        # mask low score keypoints
        score_result[score_result < cfg['score_thr']] = 0
        score_result = score_result.sum(axis=-1)
        m = (score_result > 0).sum(axis=-1)
        rescored_score = np.divide(score_result, m, out=np.zeros_like(score_result), where=m != 0)
        # save result
        for i in range(len(kps_result)):
            result = dict(image_id=image_id[i].item(), category_id=1, score=float(round(rescored_score[i], 4)),
                          keypoints=kps_result[i].round(3).tolist())

            dump_results.append(result)

    return dump_results


def evaluation(result, coco_val, result_dir):
    result_path = osp.join(result_dir, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f)

    result = coco_val.loadRes(result_path)
    cocoEval = COCOeval(coco_val, result, iouType='keypoints')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    result_path = osp.join(result_dir, 'result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(cocoEval, f, 2)
        print("Saved result file to " + result_path)


def benchmark_model(model, n_runs=200):
    inputs = model.inputs
    if isinstance(inputs, tf.Tensor):
        inputs = [tf.Tensor]
    fake_inputs = []
    for inp in inputs:
        batched_input_shape = inp.shape.as_list()
        batched_input_shape[0] = 1
        fake_input = tf.random.uniform(batched_input_shape)
        fake_inputs.append(fake_input)

    # run model on cpu to emulate slower device
    predict_times = []  # milliseconds
    with tf.device('/cpu:0'):
        # prepare device
        outputs = model.predict(fake_inputs)
        for _ in tqdm(range(n_runs)):
            start_time = time.time()
            outputs = model.predict(fake_inputs)
            elapsed_time = time.time() - start_time
            predict_times.append(elapsed_time * 1000)
    avg = np.mean(predict_times)
    std = np.std(predict_times)
    print(f'Average inference time (2std) = {avg:.1f} ms +/- {std * 2:.1f} ms')
    return predict_times


def benchmark_tflite_model(tflite_model, n_runs=100):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    fake_input_data = np.random.rand(*input_shape).astype(np.float32)
    # print(fake_input_data.shape, fake_input_data.dtype)
    predict_times = []  # milliseconds
    for _ in tqdm(range(n_runs)):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], fake_input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        elapsed_time = time.time() - start_time
        predict_times.append(elapsed_time * 1000)

    avg = np.mean(predict_times)
    std = np.std(predict_times)
    print(f'Average inference time (2std) = {avg:.1f} ms +/- {std * 2:.1f} ms')
    return predict_times
