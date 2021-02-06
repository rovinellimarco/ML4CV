import os
import zipfile
from pathlib import Path
import os.path as osp

import wget
from tqdm.notebook import tqdm
from pycocotools.coco import COCO
import numpy as np

def download_coco_annotations(path):
    wget.download('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                  out=path + 'annotations_trainval2017.zip')
    with zipfile.ZipFile(path + 'annotations_trainval2017.zip', 'r') as zip_ref:
        zip_ref.extractall(path)
    os.remove(path + 'annotations_trainval2017.zip')


def download_images(path, data_type):
    assert data_type in ['train2017', 'val2017']
    pbar = None

    def bar_progress(current, total, width=80):
        nonlocal pbar
        units = 1024 * 1024
        if pbar is None:
            pbar = tqdm(total=total // units)
            pbar.n = 0
            pbar.refresh(nolock=True)
        if (current - pbar.n * units) / total > 0.01:
            pbar.n = current // units
            pbar.refresh(nolock=True)

    ann_filepath = f'{path}annotations/person_keypoints_{data_type}.json'
    coco = COCO(ann_filepath)
    downloaded_filepath = f'{path}{data_type}.zip'

    if not os.path.isfile(downloaded_filepath):
        wget.download(f'http://images.cocodataset.org/zips/{data_type}.zip',
                      out=downloaded_filepath, bar=bar_progress)
        pbar.n = pbar.total
        pbar.refresh(nolock=True)
        pbar.close()
        pbar = None

    cat_ids = coco.getCatIds(catNms=['person']);
    img_ids = coco.getImgIds(catIds=cat_ids);
    img_ids.sort()
    filenames = []
    for image_info in coco.loadImgs(img_ids):
        filenames.append(f"{data_type}/{image_info['file_name']}")
    # print(len(filenames))
    Path(path + 'images').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(downloaded_filepath, 'r') as inzipfile:
        for infile in tqdm(filenames):
            inzipfile.extract(infile, path=path + 'images/')
    os.remove(downloaded_filepath)


def get_coco(config):
    coco_val = COCO(osp.join(config['path'], 'annotations', 'person_keypoints_val2017.json'))
    coco_train = COCO(osp.join(config['path'], 'annotations', 'person_keypoints_train2017.json'))
    return coco_train, coco_val


def annotations_generator(cfg, coco, data_type):
    img_path = osp.join(cfg['path'], 'images', data_type)
    # generate all annotations for one image in order to lower read limit speed
    person_img_ids = coco.getImgIds(catIds=[1])
    for img_id in person_img_ids:
        img_filename = coco.imgs[img_id]['file_name']
        image_filepath = osp.join(img_path, img_filename)
        annotations_ids = coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=cfg['crowds'])

        annotations = []
        bboxes = []
        for i, ann_id in enumerate(annotations_ids):
            ann = coco.anns[ann_id]
            joints = ann['keypoints']
            # skip images with no visible keypoints
            if (np.sum(joints[2::3]) == 0) or (ann['num_keypoints'] == 0):  # ann['image_id'] not in coco.imgs)
                continue

            # sanitize bboxes
            x, y, w, h = ann['bbox']
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
            else:
                continue

            # joints
            annotations.append([joints[i:i + 3] for i in range(0, len(joints), 3)])
        if len(bboxes) == 0:
            continue
        yield img_id, image_filepath, np.array(bboxes, dtype=np.float32), annotations