import copy
import logging
import os.path as osp
import random
from collections import OrderedDict
from collections import defaultdict

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from tabulate import tabulate
from termcolor import colored

from datasets.process import get_affine_transform, fliplr_joints, exec_affine_transform, generate_heatmaps, \
    half_body_transform, convert_data_to_annorect_struct, dark_get_affine_transform
from datasets.transforms import build_transforms
from datasets.zoo.base import VideoDataset
from engine.defaults.constant import DATASET_REGISTRY
from utils.common import TRAIN_PHASE, VAL_PHASE
from utils.utils_bbox import box2cs
from utils.utils_folder import create_folder
from utils.utils_image import read_image
from utils.utils_json import read_json_from_file, write_json_to_file
from .posetrack_utils import video2filenames, evaluate_simple

from thirdparty.clustering import k_means


@DATASET_REGISTRY.register()
class PoseTrack_Alignment(VideoDataset):
    """
        PoseTrack_Alignment
    """

    def __init__(self, cfg, phase, **kwargs):
        super(PoseTrack_Alignment, self).__init__(cfg, phase, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.train = True if phase == TRAIN_PHASE else False
        self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.joints_weight = np.array(
            [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
            dtype=np.float32).reshape((self.num_joints, 1))
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.is_posetrack18 = cfg.DATASET.IS_POSETRACK18
        self.transform = build_transforms(cfg, phase)

        # self.distance_whole_otherwise_segment = cfg.DISTANCE_WHOLE_OTHERWISE_SEGMENT
        self.distance = cfg.DISTANCE
        # self.previous_distance = cfg.PREVIOUS_DISTANCE
        # self.next_distance = cfg.NEXT_DISTANCE

        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME

        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        self.sigma = cfg.MODEL.SIGMA

        self.img_dir = cfg.DATASET.IMG_DIR
        self.json_dir = cfg.DATASET.JSON_DIR
        self.test_on_train = cfg.DATASET.TEST_ON_TRAIN
        self.json_file = cfg.DATASET.JSON_FILE

        if self.phase != TRAIN_PHASE:
            self.img_dir = cfg.DATASET.TEST_IMG_DIR
            temp_subCfgNode = cfg.VAL if self.phase == VAL_PHASE else cfg.TEST
            self.nms_thre = temp_subCfgNode.NMS_THRE
            self.image_thre = temp_subCfgNode.IMAGE_THRE
            self.soft_nms = temp_subCfgNode.SOFT_NMS
            self.oks_thre = temp_subCfgNode.OKS_THRE
            self.in_vis_thre = temp_subCfgNode.IN_VIS_THRE
            self.bbox_file = temp_subCfgNode.COCO_BBOX_FILE
            self.use_gt_bbox = temp_subCfgNode.USE_GT_BBOX
            self.annotation_dir = temp_subCfgNode.ANNOT_DIR

        self.coco = COCO(
            osp.join(self.json_dir, 'posetrack_train.json' if self.is_train else 'posetrack_val.json'))
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)

        self.video_image_index = None

        self.clustering = True
        if self.clustering:
            clustering_data = self._load_normalization_coco_keypoints_annotations()
            write_json_to_file(clustering_data.tolist(),
                               osp.join('/media/Z/frunyang/FAMI-Pose/thirdparty/clustering/pose_analysis',
                                        '17val.json'))

        self.data = self._list_data()

        vis_data_gt = False
        if vis_data_gt:
            self.img_pose_data = self._to_img_pose()
            print("=== Start Vis PoseTrac Dataset ===")
            self._vis_hook('/media/Z/frunyang/PoseTrack_Visualization/GT')
            print("=== Successfully Writing! ===")
            import sys
            sys.exit("=== Endding ===")

        self.model_input_type = cfg.DATASET.INPUT_TYPE  # ['single_frame','single_ref_frame','spatiotemporal_window']

        self.show_data_parameters()
        self.show_samples()

        self.number_sup = cfg.NUMBER_SUP

    def __getitem__(self, key_item_index):
        data_item = copy.deepcopy(self.data[key_item_index])
        if self.model_input_type == "single_ref_frame":
            input_x, ref_x, target_heatmaps, target_heatmaps_weight, meta = self._get_single_ref(data_item)
            neighbor_ref_x = torch.cat(ref_x, dim=0)
        else:
            input_x, ref_x, target_heatmaps, target_heatmaps_weight, meta = self._get_spatiotemporal_window(
                data_item)
            neighbor_ref_x = torch.cat(ref_x, dim=0)

        return input_x, neighbor_ref_x, target_heatmaps, target_heatmaps_weight, meta

    def _get_single_ref(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        if zero_fill == 6:
            is_posetrack18 = True
        else:
            is_posetrack18 = False

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        # if self.distance_whole_otherwise_segment:
        farthest_distance = self.distance

        prev_delta_range = range(1,
                                 min((current_idx + 1) if is_posetrack18 else current_idx, farthest_distance))
        next_delta_range = range(1, min((num_frames - current_idx) if is_posetrack18 else (
                num_frames - current_idx + 1), farthest_distance))

        prev_delta_range = list(prev_delta_range)
        while len(prev_delta_range) < (farthest_distance - 1):
            prev_delta_range.append(0)

        next_delta_range = list(next_delta_range)
        while len(next_delta_range) < (farthest_distance - 1):
            next_delta_range.append(0)

        prev_delta_range = [x * -1 for x in prev_delta_range]
        ref_delta_range = prev_delta_range + next_delta_range

        ids = OrderedDict()
        ids["supp_idx"] = []
        ids["current_idx"] = current_idx

        ref_image_files = []

        delta_list = random.sample(ref_delta_range, self.number_sup)
        for delta in delta_list:
            ref_input_id = current_idx + delta
            ref_image_file = osp.join(osp.dirname(image_file_path),
                                      str(ref_input_id).zfill(zero_fill) + '.jpg')
            ref_image_files.append(ref_image_file)
            ids["supp_idx"].append(ref_input_id)

        ref_data_numpy_list = []

        for ref_image_file in ref_image_files:
            if not osp.exists(ref_image_file):
                error_msg = "Can not find image :{}".format(ref_image_file)
                self.logger.error(error_msg)
                raise Exception(error_msg)
            ref_data_numpy = read_image(ref_image_file)
            if self.color_rgb:
                # cv2 read_image  color channel is BGR
                ref_data_numpy = cv2.cvtColor(ref_data_numpy, cv2.COLOR_BGR2RGB)
            if ref_data_numpy is None:
                self.logger.error('=> fail to read {}'.format(ref_image_file))
                raise ValueError('Fail to read {}'.format(ref_image_file))
            ref_data_numpy_list.append(ref_data_numpy)

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:,
                       0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints,
                                                               self.upper_body_ids,
                                                               self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]

                for i in range(len(ref_data_numpy_list)):
                    ref_data_numpy_list[i] = ref_data_numpy_list[i][:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        # trans = dark_get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                 flags=cv2.INTER_LINEAR)

        for i in range(len(ref_data_numpy_list)):
            ref_data_numpy_list[i] = cv2.warpAffine(ref_data_numpy_list[i], trans,
                                                    (int(self.image_size[0]), int(self.image_size[1])),
                                                    flags=cv2.INTER_LINEAR)

        '''
        test -- input data visualized
        '''
        # path = '/media/Z/frunyang/global/test/visual'
        # for index, img_data in enumerate(ref_data_numpy_list):
        #     cv2.imwrite(path + '/' + str(index) + '.jpg', img_data)
        # cv2.imwrite(path + '/' + '1a' + '.jpg', input_x)

        if self.transform:
            input_x = self.transform(input_x)
            for i in range(len(ref_data_numpy_list)):
                ref_data_numpy_list[i] = self.transform(ref_data_numpy_list[i])

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma,
                                                                    self.image_size,
                                                                    self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta = {
            'image': image_file_path,
            'ref_sup_image': ref_image_files,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
            'frame_id': ids
        }

        return input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta

    def _get_spatiotemporal_window(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        if zero_fill == 6:
            is_posetrack18 = True
        else:
            is_posetrack18 = False

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        # if self.distance_whole_otherwise_segment:
        farthest_distance = self.distance

        prev_delta_range = range(1,
                                 min((current_idx + 1) if is_posetrack18 else current_idx, farthest_distance))
        next_delta_range = range(1, min((num_frames - current_idx) if is_posetrack18 else (
                num_frames - current_idx + 1), farthest_distance))

        prev_delta_range = list(prev_delta_range)
        next_delta_range = list(next_delta_range)

        prev_delta_sup_range, next_delta_sup_range = [], []
        if len(prev_delta_sup_range) == 0:
            for i in range(self.distance - 1 - len(prev_delta_range)):
                prev_delta_sup_range.append(0)
        else:
            for i in range(self.distance - 1 - len(prev_delta_range)):
                prev_delta_sup_range.append(1)

        if len(next_delta_range) == 0:
            for i in range(self.distance - 1 - len(next_delta_range)):
                next_delta_sup_range.append(0)
        else:
            for i in range(self.distance - 1 - len(next_delta_range)):
                next_delta_sup_range.append(1)

        prev_delta_sup_range.extend(prev_delta_range)
        prev_delta_sup_range.reverse()
        prev_delta_range = prev_delta_sup_range
        next_delta_range.extend(next_delta_sup_range)

        ids = OrderedDict()
        ids["supp_idx"] = []
        ids["current_idx"] = current_idx

        ref_image_files = []
        for prev_delta in prev_delta_range:
            prev_input_id = current_idx - prev_delta
            prev_image_file = osp.join(osp.dirname(image_file_path),
                                       str(prev_input_id).zfill(zero_fill) + '.jpg')
            ref_image_files.append(prev_image_file)
            ids["supp_idx"].append(prev_input_id)

        for next_delta in next_delta_range:
            next_input_id = current_idx + next_delta
            next_image_file = osp.join(osp.dirname(image_file_path),
                                       str(next_input_id).zfill(zero_fill) + '.jpg')
            ref_image_files.append(next_image_file)
            ids["supp_idx"].append(next_input_id)

        ref_data_numpy_list = []

        for ref_image_file in ref_image_files:
            if not osp.exists(ref_image_file):
                error_msg = "Can not find image :{}".format(prev_image_file)
                self.logger.error(error_msg)
                raise Exception(error_msg)
            ref_data_numpy = read_image(ref_image_file)
            if self.color_rgb:
                # cv2 read_image  color channel is BGR
                ref_data_numpy = cv2.cvtColor(ref_data_numpy, cv2.COLOR_BGR2RGB)
            if ref_data_numpy is None:
                self.logger.error('=> fail to read {}'.format(ref_image_file))
                raise ValueError('Fail to read {}'.format(ref_image_file))
            ref_data_numpy_list.append(ref_data_numpy)

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:,
                       0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints,
                                                               self.upper_body_ids,
                                                               self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]

                for i in range(len(ref_data_numpy_list)):
                    ref_data_numpy_list[i] = ref_data_numpy_list[i][:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        # trans = get_affine_transform(center, scale, r, self.image_size)
        trans = dark_get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                 flags=cv2.INTER_LINEAR)

        for i in range(len(ref_data_numpy_list)):
            ref_data_numpy_list[i] = cv2.warpAffine(ref_data_numpy_list[i], trans,
                                                    (int(self.image_size[0]), int(self.image_size[1])),
                                                    flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            for i in range(len(ref_data_numpy_list)):
                ref_data_numpy_list[i] = self.transform(ref_data_numpy_list[i])

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma,
                                                                    self.image_size,
                                                                    self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta = {
            'image': image_file_path,
            'ref_sup_image': ref_image_files,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
            'frame_id': ids
        }

        return input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta

    def _get_global_ref_images(self, key_item_index):
        """
            方法1. 给出一个时间范围，在该时间范围内均匀的找 ref image.

            return : 返回图片数据
        """
        num_samples = 5
        options_mode = ['only_key_frame', 'exclude_key_frame', 'total_video']
        ref_frame_bbox_conf_threshold = 0.6

        key_data_item = self.data[key_item_index]
        video_name = key_data_item['video_name']
        key_frame_id = key_data_item['frame_id']

        mode = 'exclude_key_frame'
        if mode == 'only_key_frame':
            valid_ref_frame_ids = [key_frame_id]
        elif mode == 'exclude_key_frame':
            valid_ref_frame_ids = list(self.video_image_index[video_name].keys())
            valid_ref_frame_ids.remove(key_frame_id)
        elif mode == 'total_video':
            valid_ref_frame_ids = list(self.video_image_index[video_name].keys())
        else:
            raise Exception("model error")

        candidate_item_ids = []
        for valid_ref_frame_id in valid_ref_frame_ids:
            ref_item_ids = self.video_image_index[video_name][valid_ref_frame_id]
            assert len(ref_item_ids) > 0
            candidate_item_ids.extend(ref_item_ids)

        # 添加筛选
        valid_item_ids = []
        for item_id in candidate_item_ids:
            item = self.data[item_id]
            if "score" not in item:
                valid_item_ids = candidate_item_ids
                break
            else:
                if item['score'] > ref_frame_bbox_conf_threshold:
                    valid_item_ids.append(item_id)

        ref_item_ids = random.sample(valid_item_ids, min(len(valid_item_ids), num_samples))
        if len(ref_item_ids) < num_samples:
            # 如果数量不够,重复采样,凑够个数
            ref_item_ids.extend(random.sample(valid_item_ids, num_samples - len(ref_item_ids)))

        ref_x_list = []

        # path = '/media/Z/frunyang/global/test/visual'

        for ref_item_id in ref_item_ids:
            ref_data_item = self.data[ref_item_id]
            image_file_path = ref_data_item['image']
            data_numpy = read_image(image_file_path)
            if self.color_rgb:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

            center, scale, r = ref_data_item["center"], ref_data_item["scale"], 0

            trans = get_affine_transform(center, scale, r, self.image_size)
            ref_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                   flags=cv2.INTER_LINEAR)
            # cv2.imwrite(path + '/gloablref_' + str(ref_item_id) + '.jpg',ref_x)
            if self.transform:
                ref_x = self.transform(ref_x)
            ref_x_list.append(ref_x)

        # for idx, ref_x in enumerate(ref_x_list):
        #     cv2.imwrite(path + '/' + 'global_' + str(idx) + '.jpg',ref_x)

        # 都拼到通道上了
        ref_images = torch.cat(ref_x_list, dim=0)

        return ref_images

    def _get_single_frame(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)
        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))
        ids = OrderedDict()
        ids["supp_idx"] = []
        ids["current_idx"] = current_idx

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:,
                       0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints,
                                                               self.upper_body_ids,
                                                               self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        # trans = get_affine_transform(center, scale, r, self.image_size)
        trans = dark_get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                 flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

            # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma,
                                                                    self.image_size,
                                                                    self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta = {
            'image': image_file_path,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
            'frame_id': ids
        }

        return input_x, target_heatmaps, target_heatmaps_weight, meta

    def _list_data(self):
        if self.is_train or self.use_gt_bbox:
            # use bbox from annotation
            data = self._load_coco_keypoints_annotations()
        else:
            # use bbox from detection
            data = self._load_detection_results()

        # 记载视频
        video_image_index = dict()
        for item_index, item in enumerate(data):
            video_name = item['video_name']
            frame_id = item['frame_id']
            # if video_name not in
            if video_name not in video_image_index.keys():
                video_image_index[video_name] = defaultdict(list)
            video_image_index[video_name][frame_id].append(item_index)

        self.video_image_index = video_image_index

        return data

    def _load_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """
        gt_db = []
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []

            for obj in objs:
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)

                rec.append({
                    'image': osp.join(self.img_dir, file_name),
                    'center': center,
                    'scale': scale,
                    'box': obj['clean_bbox'][:4],
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'nframes': nframes,
                    'frame_id': frame_id,
                    'video_name': "/".join(file_name.split('/')[:-1]),
                })

            gt_db.extend(rec)
        return gt_db

    def _load_normalization_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """
        clustering_joints = []
        gt_db = []
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []

            for obj in objs:
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)

                rec.append({
                    'image': osp.join(self.img_dir, file_name),
                    'center': center,
                    'scale': scale,
                    'box': obj['clean_bbox'][:4],
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'nframes': nframes,
                    'frame_id': frame_id,
                    'video_name': "/".join(file_name.split('/')[:-1]),
                })

                # === Clustering Data ===
                if self.clustering:
                    r = 0
                    trans = dark_get_affine_transform(center, scale, r, self.image_size)

                    joints = copy.deepcopy(joints_3d)

                    # joint transform like image
                    for i in range(self.num_joints):
                        if joints_3d_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                            joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)
                    clustering_joints.append(np.expand_dims(joints[:, :2], 0))

            gt_db.extend(rec)
        assert len(gt_db) == len(clustering_joints), 'Length Error!'
        clustering_joints = np.concatenate(clustering_joints, 0)
        return clustering_joints

    def _load_detection_results(self):
        logger = logging.getLogger(__name__)
        logger.info("=> Load bbox file from {}".format(self.bbox_file))
        all_boxes = read_json_from_file(self.bbox_file)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_data = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']  # xywh
            score = det_res['score']
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = box2cs(box, self.aspect_ratio, self.bbox_enlarge_factor)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_data.append({
                'image': osp.join(self.img_dir, img_name),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'nframes': nframes,
                'frame_id': frame_id,
                'video_name': "/".join(img_name.split('/')[:-1]),
            })
        # logger.info('=> Total boxes: {}'.format(len(all_boxes)))
        # logger.info('=> Total boxes after filter low score@{}: {}'.format(self.image_thre, num_boxes))

        table_header = ["Total boxes", "Filter threshold", "Remaining boxes"]
        table_data = [[len(all_boxes), self.image_thre, num_boxes]]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Boxes Info Table: \n" + colored(table, "magenta"))

        return kpt_data

    def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        ### processing our preds
        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        all_tracks = []
        cc = 0

        # print(img_path)
        for key in img_path:
            temp = key.split('/')

            # video_name = osp.dirname(key)
            video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
            img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

            prev_nm = temp[len(temp) - 1]
            frame_num = int(prev_nm.replace('.jpg', ''))
            if not video_name in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            idx_list = img_path[key]
            pose_list = []
            box_list = []
            for idx in idx_list:
                temp = np.zeros((4, 17))
                temp[0, :] = preds[idx, :, 0]
                temp[1, :] = preds[idx, :, 1]
                temp[2, :] = preds[idx, :, 2]
                temp[3, :] = preds[idx, :, 2]
                pose_list.append(temp)

                temp = np.zeros((1, 6))
                temp[0, :] = boxes[idx, :]
                box_list.append(temp)

            all_preds.append(pose_list)
            all_boxes.append(box_list)
            cc += 1

        annot_dir = self.annotation_dir
        is_posetrack18 = self.is_posetrack18

        out_data = {}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            c = 0
            used_frame_list = []
            cur_length = L['images/' + vid]

            temp_kps_map = {}
            temp_track_kps_map = {}
            temp_box_map = {}

            for idx in idx_list:
                frame_num = vid2frame_map[vid][c]
                img_sfx = vid2name_map[vid][c]
                c += 1

                used_frame_list.append(frame_num)

                kps = all_preds[idx]
                temp_kps_map[frame_num] = (img_sfx, kps)

                bb = all_boxes[idx]
                temp_box_map[frame_num] = bb
            #### including empty frames
            nnz_counter = 0
            next_track_id = 0

            if not is_posetrack18:
                sid = 1
                fid = cur_length + 1
            else:
                sid = 0
                fid = cur_length
            # start id ~ finish id
            for current_frame_id in range(sid, fid):
                frame_num = current_frame_id
                if not current_frame_id in used_frame_list:
                    temp_sfx = vid2name_map[vid][0]
                    arr = temp_sfx.split('/')
                    if not is_posetrack18:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(8) + '.jpg'
                    else:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(6) + '.jpg'
                    kps = []
                    tracks = []
                    bboxs = []

                else:

                    img_sfx = temp_kps_map[frame_num][0]
                    kps = temp_kps_map[frame_num][1]
                    bboxs = temp_box_map[frame_num]
                    tracks = [track_id for track_id in range(len(kps))]
                    # tracks = [1] * len(kps)

                ### creating a data element
                data_el = {
                    'image': {'name': img_sfx},
                    'imgnum': [frame_num],
                    'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs),
                }
                if vid in out_data:
                    out_data[vid].append(data_el)
                else:
                    out_data[vid] = [data_el]

        logger.info("=> saving files for evaluation")
        #### saving files for evaluation
        for vname in out_data:
            vdata = out_data[vname]
            outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        # run evaluation
        # AP = self._run_eval(annot_dir, output_dir)[0]
        AP = evaluate_simple.evaluate(annot_dir, output_dir, eval_track=False)[0]

        name_value = [
            ('Head', AP[0]),
            ('Shoulder', AP[1]),
            ('Elbow', AP[2]),
            ('Wrist', AP[3]),
            ('Hip', AP[4]),
            ('Knee', AP[5]),
            ('Ankle', AP[6]),
            ('Mean', AP[7])
        ]

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

    def _to_img_pose(self):
        all_data = copy.deepcopy(self.data)
        img_box = OrderedDict()

        for data in all_data:
            img = data["image"]
            box = data["box"]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            joints = data["joints_3d"]
            if img not in img_box.keys():
                img_box[img] = dict()
                img_box[img]['box'] = []
                img_box[img]['pose'] = []
                img_box[img]['box'].append(box)
                img_box[img]['pose'].append(joints)
            else:
                img_box[img]['box'].append(box)
                img_box[img]['pose'].append(joints)
        return img_box

    def _vis_hook(self, vis_output_dir):
        from engine.core.utils.vis_helper import draw_skeleton_in_origin_image
        from tqdm.auto import tqdm
        # for name, value in self.img_pose_data.items():
        for name in tqdm(self.img_pose_data.keys()):
            image = name
            value = self.img_pose_data[name]
            batch_box = np.concatenate(
                [np.expand_dims(box, 0) for box in np.array(value['box']) if True][:], 0)
            batch_pose = np.concatenate(
                [np.expand_dims(pose, 0) for pose in np.array(value['pose']) if True][:], 0)

            draw_skeleton_in_origin_image([image], batch_pose, batch_box, vis_output_dir,
                                          vis_skeleton=True, vis_bbox=True,
                                          sure_threshold=0.0)
