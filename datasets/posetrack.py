import glob
import json
import os
import tarfile

import wget as wget

from datasets.custom_tar_reader import CustomTarFileReader
import os.path as osp
import numpy as np


def download_posetrack_annotations(path):
    wget.download('https://posetrack.net/posetrack18-data/posetrack18_v0.45_public_labels.tar.gz',
                  out=path + 'annotations.tar.gz')
    with tarfile.open(path + 'annotations.tar.gz', mode='r:gz') as zip_ref:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(zip_ref, path)
    os.remove(path + 'annotations.tar.gz')


def download_posetrack_images(path, count=-1):
    if count == -1:
        count = 18
    for c in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'][:count]:
        wget.download('https://posetrack.net/posetrack18-data/posetrack18_images.tar.a' + c,
                      out=path + 'images.tar.a' + c)
    filenames = sorted(glob.glob(path + 'images.tar.*'))
    f = CustomTarFileReader(filenames)
    with tarfile.open(fileobj=f) as zip_ref:
        zip_ref.extractall(path)
    # os.remove(path + 'images.tar.gz')


def posetrack_annotations_generator(cfg, data_type, only_existing_folders=True):
    ann_path = osp.join(cfg['path'], 'posetrack_data', 'annotations', data_type, '*.json')
    def_kipid2name = None  # for consistency checks
    file_track_ids = {}
    for mat_filename in glob.glob(ann_path):
        start_track_ids = None
        # shape (1, n_frames)
        json_data = open(mat_filename).read()
        anns = json.loads(json_data)
        # images
        imgid2imgname = {}
        for image in anns['images']:
            imgid2imgname[image['id']] = image['file_name']
            # is labeled ecc..7

        # categories
        kpidx2name = {}
        for cat in anns['categories']:
            for idx, kpname in enumerate(cat['keypoints']):
                kpidx2name[idx] = kpname

        # checking all keypoints have the same order in all files
        current_kipid2name = [kpidx2name[i] for i in range(len(kpidx2name))]
        if def_kipid2name is None:
            def_kipid2name = current_kipid2name
        else:
            assert all([def_kipid2name[i] == current_kipid2name[i] for i in range(len(def_kipid2name))])
        # annotations
        imgid_track_to_kps = {}
        track_ids = []
        discarded_ids = set()
        last_track_id = None
        last_image_id = None
        frames = set()
        for person_ann in anns['annotations']:
            track_id = person_ann['track_id']
            image_id = person_ann['image_id']
            frames.add(image_id)
            if last_track_id is None:
                last_track_id = track_id
                last_image_id = image_id
            else:
                assert (
                                   track_id > last_track_id and last_image_id == image_id) or last_image_id < image_id, f'{track_id} {last_track_id} {image_id} {last_image_id}'

            track_ids.append(track_id)

        track_ids = sorted(track_ids)
        unique_track_ids = set(track_ids)

        file_track_ids[mat_filename] = unique_track_ids  # track_ids

    # return
    ###########################################################

    for mat_filename in glob.glob(ann_path):
        json_data = open(mat_filename).read()
        anns = json.loads(json_data)

        # image_filename    # (n_frames,)
        # bboxes            # (n_frames, n_track_ids, 4) => (n_frames*n_track_ids, 4) => filter out boxes not present anymore on image
        # keypoints         # (n_frames, n_track_ids, 17, 2)
        # box_indices       # (n_frames*n_track_ids,)
        # model_input       # (n_frames, img, (17,2))

        # image_filename    # (n_frames,)
        imgid2imgname = {}
        for image in anns['images']:
            imgid2imgname[image['id']] = image['file_name']

        n_track_ids = len(file_track_ids[mat_filename])
        track2id = {id: real_id for real_id, id in enumerate(file_track_ids[mat_filename])}
        imageids = []
        bboxes = []
        keypoints = []
        last_image_id = None
        for person_ann in anns['annotations']:
            if person_ann['track_id'] not in track2id:
                continue
            track_id = track2id[person_ann['track_id']]
            image_id = person_ann['image_id']

            if last_image_id != image_id:
                last_image_id = image_id
                imageids.append(image_id)
                bboxes.append([[0, 0, 0, 0] for _ in range(n_track_ids)])
                keypoints.append([[[0, 0, 0] for z in range(17)] for _ in range(n_track_ids)])
                cur_frame_bboxes = bboxes[-1]
                cur_frame_kps = keypoints[-1]

            joints = person_ann['keypoints']
            if np.sum(joints[2::3]) == 0:
                continue

            cur_bbox = person_ann['bbox']  # person_ann['bbox_head']
            x1, y1, x2, y2 = cur_bbox
            x2 = x1 + x2  # truth is, they are the width
            y2 = y1 + y2  # and the height
            cur_bbox = [x1, y1, x2 - x1, y2 - y1]

            cur_frame_bboxes[track_id] = cur_bbox
            kps = person_ann['keypoints']
            cur_frame_kps[track_id] = [kps[i:i + 3] for i in range(0, len(kps), 3)]

        # FOR MATFILE
        image_filepaths = [imgid2imgname[id] for id in imageids]
        bboxes = np.array(bboxes, dtype=np.float32)
        try:
            keypoints = np.array(keypoints, dtype=np.float32)
        except:
            for k in keypoints:
                for c in k:
                    print(len(c))
        assert len(imageids) == len(set(imageids))

        if only_existing_folders:
            if not osp.exists(image_filepaths[0]):
                continue

        for t_id in range(bboxes.shape[1]):
            f1 = (bboxes[:, t_id, :].sum(axis=-1) > 0)  # n_frames, 1

            r1 = np.zeros_like(f1)

            cumulative_and = f1[cfg['min_frames'] - 1:].copy()  # last_history_frame
            for history_idx in range(0, cfg['min_frames'] - 1):
                cumulative_and &= f1[history_idx: history_idx - cfg['min_frames'] + 1]
            r1[cfg['min_frames'] - 1:] = cumulative_and

            if r1.sum() == 0:
                continue
            history_mask = r1
            filtered_bboxes = bboxes[history_mask, t_id, :]
            filtered_keypoints = keypoints[history_mask, t_id, :, :]
            imageids = np.array(imageids)
            image_filepaths = np.array(image_filepaths)
            filtered_imageids = imageids[history_mask]
            filtered_image_filepaths = image_filepaths[history_mask]
            yield filtered_imageids, filtered_image_filepaths, filtered_bboxes, filtered_keypoints
