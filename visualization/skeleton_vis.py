#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/11/25
    Description:
"""
import cv2

from datasets.zoo.posetrack.pose_skeleton import PoseTrack_COCO_Keypoint_Ordering, PoseTrack_Official_Keypoint_Ordering, \
    PoseTrack_Keypoint_Pairs
# from datasets.zoo.posetrack import PoseTrack_Keypoint_Pairs, PoseTrack_Official_Keypoint_Ordering, PoseTrack_COCO_Keypoint_Ordering
# from datasets.zoo.posetrack import PoseTrack_Keypoint_Pairs, PoseTrack_Official_Keypoint_Ordering, PoseTrack_COCO_Keypoint_Ordering
# from datasets.zoo.HiEve import HiEve_Keypoint_Pairs, HiEve_Official_Keypoint_Ordering, HiEve_COCO_Keypoint_Ordering
from utils.utils_color import COLOR_DICT


# def draw_skeleton_in_origin_image(batch_image_list, batch_joints_list, batch_bbox_list, save_dir, vis_skeleton=True, vis_bbox=True):
#     """
#     :param batch_image_list:  batch image path
#     :param batch_joints_list:   joints coordinates in image Coordinate reference system
#     :batch_bbox_list: xyxy
#     :param save_dir:
#     :return: No return
#     """
#
#     skeleton_image_save_folder = osp.join(save_dir, "skeleton")
#     bbox_image_save_folder = osp.join(save_dir, "bbox")
#     together_save_folder = osp.join(save_dir, "SkeletonAndBbox")
#
#     if vis_skeleton and vis_bbox:
#         save_folder = together_save_folder
#     else:
#         save_folder = skeleton_image_save_folder
#         if vis_bbox:
#             save_folder = bbox_image_save_folder
#
#     batch_final_coords = batch_joints_list
#
#     for index, image_path in enumerate(batch_image_list):
#         final_coords = batch_final_coords[index]
#         final_coords = coco2posetrack_ord_infer(final_coords)
#         bbox = batch_bbox_list[index]
#
#         image_name = image_path[image_path.index("images") + len("images") + 1:]
#
#         vis_image_save_path = osp.join(save_folder, image_name)
#         if osp.exists(vis_image_save_path):
#             processed_image = read_image(vis_image_save_path)
#
#             processed_image = draw_skeleton(processed_image, final_coords, threshold=0.2, joint_ordering="POSETRACK_OFFICIAL",
#                                             vis_joint_label=True) if vis_skeleton else processed_image
#
#             # processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
#             #                                                           flag_only_draw_sure=True) if vis_skeleton else processed_image
#
#             processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image
#
#             save_image(vis_image_save_path, processed_image)
#         else:
#             image_data = read_image(image_path)
#             processed_image = image_data.copy()
#
#             processed_image = draw_skeleton(processed_image, final_coords, threshold=0.2, joint_ordering="POSETRACK_OFFICIAL",
#                                             vis_joint_label=True) if vis_skeleton else processed_image
#             # processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
#             #                                                           flag_only_draw_sure=True) if vis_skeleton else processed_image
#
#             processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image
#
#             save_image(vis_image_save_path, processed_image)


# def draw_skeleton_from_tensor(batch_img_tensor, batch_heatmaps_tensor):
#     joints, joints_conf = get_max_preds(batch_heatmaps_tensor.cpu().detach().numpy())
#     joints_coords = np.concatenate([joints * 4, joints_conf], axis=-1)
#     batch_size = batch_img_tensor.shape[0]
#
#     skeleton_images = []
#     for batch_index in range(batch_size):
#         bbox_image = tensor2im(batch_img_tensor[batch_index])
#         skeleton_image = draw_skeleton(bbox_image.copy(), joints_coords[batch_index], threshold=0.2)
#         skeleton_images.append(skeleton_image)
#
#     return skeleton_images


def draw_skeleton(img, joints, threshold=0.8, joint_ordering="posetrack_coco", vis_joint_label=False):
    """

    :param img:  data
    :param joints:
    :param threshold:
    :param joint_ordering:
    :param vis_joint_label:
    :return:
    """
    joint_ordering = joint_ordering.upper()
    if joint_ordering == "POSETRACK_OFFICIAL":
        Keypoint_Pairs = PoseTrack_Keypoint_Pairs
        Keypoint_Ordering = PoseTrack_Official_Keypoint_Ordering
    elif joint_ordering == "POSETRACK_COCO":
        Keypoint_Pairs = PoseTrack_Keypoint_Pairs
        Keypoint_Ordering = PoseTrack_COCO_Keypoint_Ordering
    else:
        raise Exception("Undefined joint ordering {}".format(joint_ordering))

    for joint_pair in Keypoint_Pairs:
        ind_1 = Keypoint_Ordering.index(joint_pair[0])
        ind_2 = Keypoint_Ordering.index(joint_pair[1])

        color = COLOR_DICT[joint_pair[2]]

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 1 and y1 <= 1:
            continue
        if x2 <= 1 and y2 <= 1:
            continue

        if sure1 >= threshold and sure2 >= threshold:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=6)

    if vis_joint_label:

        font_thickness = 1
        font_scale = 0.5
        t_size = cv2.getTextSize('0.22', 0, fontScale=font_scale, thickness=font_thickness)[0]

        for joint in joints:
            x, y, c = joint
            c = round(c, 2)
            x = x - t_size[0] / 2
            y = y - t_size[1] / 2
            if c >= threshold:
                cv2.putText(img, str(c), (int(x), int(y)), cv2.FONT_ITALIC, font_scale, (0, 0, 255), font_thickness)

    return img


#
# def add_poseTrack_joint_connection_to_image(img_demo, joints, sure_threshold=0.8, flag_only_draw_sure=False, vis_joint_conf=False):
#     for joint_pair in PoseTrack_Keypoint_Pairs:
#         ind_1 = PoseTrack_Official_Keypoint_Ordering.index(joint_pair[0])
#         ind_2 = PoseTrack_Official_Keypoint_Ordering.index(joint_pair[1])
#
#         color = COLOR_DICT[joint_pair[2]]
#
#         x1, y1, sure1 = joints[ind_1]
#         x2, y2, sure2 = joints[ind_2]
#
#         if x1 <= 5 and y1 <= 5: continue
#         if x2 <= 5 and y2 <= 5: continue
#
#         if flag_only_draw_sure is False:
#             sure1 = sure2 = 1
#         if sure1 > sure_threshold and sure2 > sure_threshold:
#             cv2.line(img_demo, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=6)
#     return img_demo


def circle_vis_point(img, joints):
    for joint in joints:
        x, y, c = [int(i) for i in joint]
        cv2.circle(img, (x, y), 3, (255, 255, 255), thickness=3)

    return img
