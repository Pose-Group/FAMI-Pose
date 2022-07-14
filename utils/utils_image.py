#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/01
    Description:
"""
import os
import os.path as osp

import cv2
import numpy as np
import torch

from .utils_folder import create_folder, folder_exists


def read_image(image_path):
    if not osp.exists(image_path):
        raise Exception("image_path:{},读取失败，请检测路径".format(image_path))
    img = cv2.imread(image_path)
    # image_byte_array = open(image_path, 'rb').read()
    # if image_byte_array is None:
    #     raise Exception("image_path:{},读取失败，请检测路径".format(image_path))
    # img_array = np.asarray(bytearray(image_byte_array), dtype=np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return img


def save_image(image_save_path, image_data):
    if isinstance(image_data,torch.Tensor):
        image_data = image_data.numpy()
        image_data = format_np_output(image_data)
    create_folder(os.path.dirname(image_save_path))
    return cv2.imwrite(image_save_path, image_data, [100])


def format_np_output(np_array):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        np_array (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_array.shape) == 2:
        np_array = np.expand_dims(np_array, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_array.shape[0] == 1:
        np_array = np.repeat(np_array, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_array.shape[0] == 3:
        np_array = np_array.transpose([1, 2, 0])
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_array) <= 1:
        np_array = (np_array * 255).astype(np.uint8)
    return np_array


def make_images_from_video(video_path, outimages_path=None, zero_fill=8):
    cap = cv2.VideoCapture(video_path)
    isOpened = cap.isOpened()
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率<每秒中展示多少张图片>
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取宽度
    i = 0
    if outimages_path is not None:
        if not folder_exists(outimages_path):
            create_folder(outimages_path)
    assert isOpened, "Can't find video"
    for index in range(video_length):
        (flag, data) = cap.read()  # 读取每一张 flag<读取是否成功> frame<内容>
        file_name = "{}.jpg".format(str(index).zfill(zero_fill))  # start from zero
        if outimages_path is not None:
            file_path = os.path.join(outimages_path, file_name)
        else:
            create_folder("output")
            file_path = os.path.join("output", file_name)
        if flag:  # 读取成功的话
            # 写入文件，1 文件名 2 文件内容 3 质量设置
            cv2.imwrite(file_path, data, [cv2.IMWRITE_JPEG_QUALITY, 100])
