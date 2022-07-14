#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/30
    Description:
"""
import cv2
from random import random
import os.path as osp

from utils.utils_image import read_image, save_image
from datasets.process import coco2posetrack_ord_infer, coco2jhmdb_ord_infer
from datasets.zoo.posetrack import POSETRACK_joint_pairs, POSETRACK_joint, COLOR_DICT, \
    POSETRACK_joint_name_color
from datasets.zoo.jhmdb import JHMDB_Joint_Pairs, JHMDB_Keypoint_Ordering


def draw_skeleton_in_origin_image(batch_image_list, batch_joints_list, batch_bbox_list, save_dir,
                                  vis_skeleton=True, vis_bbox=True, sure_threshold=0.2):
    """
    :param batch_image_list:  batch image path
    :param batch_joints_list:   joints coordinates in image Coordinate reference system
    :batch_bbox_list: xyxy
    :param save_dir: 保存的地址
    :return: No return
    """

    skeleton_image_save_folder = osp.join(save_dir, "skeleton")
    bbox_image_save_folder = osp.join(save_dir, "bbox")
    together_save_folder = osp.join(save_dir, "SkeletonAndBbox")

    if vis_skeleton and vis_bbox:
        save_folder = together_save_folder
    else:
        save_folder = skeleton_image_save_folder
        if vis_bbox:
            save_folder = bbox_image_save_folder

    batch_final_coords = batch_joints_list

    for index, image_path in enumerate(batch_image_list):
        image_name = '/'.join(image_path.split('/')[-3:])
        vis_image_save_path = osp.join(save_folder, image_name)
        processed_image = read_image(image_path)
        for idx in range(batch_final_coords.shape[0]):
            final_coords = batch_final_coords[idx]
            final_coords = coco2posetrack_ord_infer(final_coords)
            bbox = batch_bbox_list[idx]
            # image_name = image_path[image_path.index("images") + len("images") + 1:]

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords,
                                                                      sure_threshold=sure_threshold,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image
            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

        if osp.exists(vis_image_save_path):
            save_image(vis_image_save_path, processed_image)
        else:
            processed_image = processed_image.copy()

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords,
                                                                      sure_threshold=sure_threshold,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image

            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)


def draw_jhmdb_skeleton_in_origin_image(batch_image_list, batch_joints_list, batch_bbox_list, save_dir,
                                        vis_skeleton=True, vis_bbox=True):
    """
    :param batch_image_list:  batch image path
    :param batch_joints_list:   joints coordinates in image Coordinate reference system
    :batch_bbox_list: xyxy
    :param save_dir: 保存的地址
    :return: No return
    """

    skeleton_image_save_folder = osp.join(save_dir, "skeleton")
    bbox_image_save_folder = osp.join(save_dir, "bbox")
    together_save_folder = osp.join(save_dir, "SkeletonAndBbox")

    if vis_skeleton and vis_bbox:
        save_folder = together_save_folder
    else:
        save_folder = skeleton_image_save_folder
        if vis_bbox:
            save_folder = bbox_image_save_folder

    batch_final_coords = batch_joints_list

    for index, image_path in enumerate(batch_image_list):
        final_coords = batch_final_coords[index]
        bbox = batch_bbox_list[index]

        image_name = image_path[image_path.index("Rename_Images") + len("Rename_Images") + 1:]
        vis_image_save_path = osp.join(save_folder, image_name)
        if osp.exists(vis_image_save_path):
            processed_image = read_image(vis_image_save_path)

            processed_image = add_jhmdb_joint_connection_to_image(processed_image, final_coords,
                                                                  sure_threshold=0.2,
                                                                  flag_only_draw_sure=True) if vis_skeleton else processed_image
            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)
        else:
            image_data = read_image(image_path)
            processed_image = image_data.copy()

            processed_image = add_jhmdb_joint_connection_to_image(processed_image, final_coords,
                                                                  sure_threshold=0.2,
                                                                  flag_only_draw_sure=True) if vis_skeleton else processed_image

            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)


def add_bbox_in_image(image, bbox):
    """
    :param image
    :param bbox   -  xyxy
    """

    color = (random() * 255, random() * 255, random() * 255)

    x1, y1, x2, y2 = map(int, bbox)
    image_with_bbox = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=3)
    return image_with_bbox


## 在img中,根据posetrack的关键点连接顺序连接
def add_jhmdb_joint_connection_to_image(img_demo, joints, sure_threshold=0.8, flag_only_draw_sure=False, ):
    for joint_pair in JHMDB_Joint_Pairs:
        ind_1 = JHMDB_Keypoint_Ordering.index(joint_pair[0])
        ind_2 = JHMDB_Keypoint_Ordering.index(joint_pair[1])

        color = COLOR_DICT[joint_pair[2]]

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 5 and y1 <= 5: continue
        if x2 <= 5 and y2 <= 5: continue

        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 > sure_threshold and sure2 > sure_threshold:
            # if sure1 > 0.8 and sure2 > 0.8:
            # cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=8)
            cv2.line(img_demo, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=4)
    return img_demo


def add_poseTrack_joint_connection_to_image(img_demo, joints, sure_threshold=0.8,
                                            flag_only_draw_sure=False, ):
    for joint_pair in POSETRACK_joint_pairs:
        ind_1 = POSETRACK_joint.index(joint_pair[0])
        ind_2 = POSETRACK_joint.index(joint_pair[1])

        color = COLOR_DICT[joint_pair[2]]

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 5 and y1 <= 5: continue
        if x2 <= 5 and y2 <= 5: continue

        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 >= sure_threshold and sure2 >= sure_threshold:
            # if sure1 > 0.8 and sure2 > 0.8:
            # cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=8)
            cv2.circle(img_demo, (int(x1), int(y1)), 3, (255, 255, 255), thickness=2)
            cv2.circle(img_demo, (int(x2), int(y2)), 3, (255, 255, 255), thickness=2)
            cv2.line(img_demo, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=3)
    return img_demo
