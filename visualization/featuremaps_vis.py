#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/12/24
    Description:
"""
import cv2
import torch
import numpy as np
import os.path as osp
from datasets.transforms.build import mean, std
from utils.utils_image import save_image


def save_batch_featuremaps_develop(featuremaps, save_folder, highlight=True, file_prefix=None, file_postfix=None):
    """
    :param featuremaps:  [Batch, Channels, Height, Width]
    :param save_folder:
    :param highlight:
    :param file_prefix:
    :param file_postfix:
    :return:
    """

    featuremaps = featuremaps.detach().cpu().float().numpy()
    batch_num, channel_num, _, _ = featuremaps.shape
    featuremaps = np.transpose(featuremaps, (0, 2, 3, 1))

    np.clip(featuremaps, 0, 255)

    file_prefix = "" if file_prefix is None else file_prefix + "_"
    file_postfix = "" if file_postfix is None else "_" + file_postfix
    # import random
    # channel_index = random.randint(0, channel_num-1)
    # channel_index = 99 if channel_num >= 99 else 24
    #
    # if channel_num == 1:
    #     channel_index = 0
    for batch_index in range(batch_num):
        for channel_index in range(channel_num):
            data_save_path = osp.join(save_folder, "{}{}_{}{}.jpg".format(file_prefix, batch_index, channel_index, file_postfix))
            numpy_data = featuremaps[batch_index, :, :, channel_index:channel_index + 1]
            if highlight:
                scale_factor = 255 / (np.max(numpy_data) - np.min(numpy_data) + 1e-9)
                numpy_data *= scale_factor
            save_image(data_save_path, numpy_data)


def save_batch_featuremaps(featuremaps, save_folder, highlight=True, file_prefix=None, file_postfix=None):
    """
    :param featuremaps:  [Batch, Channels, Height, Width]
    :param save_folder:
    :param highlight:
    :param file_prefix:
    :param file_postfix:
    :return:
    """
    featuremaps = featuremaps.detach().cpu().float().numpy()
    batch_num, channel_num, _, _ = featuremaps.shape
    featuremaps = np.transpose(featuremaps, (0, 2, 3, 1))

    if highlight:
        scale_factor = 255 / (np.max(featuremaps) - np.min(featuremaps) + 1e-9)
        featuremaps *= scale_factor

    np.clip(featuremaps, 0, 255)

    file_prefix = "" if file_prefix is None else file_prefix + "_"
    file_postfix = "" if file_postfix is None else "_" + file_postfix

    for batch_index in range(batch_num):
        for channel_index in range(channel_num):
            data_save_path = osp.join(save_folder, "{}{}_{}{}.jpg".format(file_prefix, batch_index, channel_index, file_postfix))
            numpy_data = featuremaps[batch_index, :, :, channel_index:channel_index + 1]
            save_image(data_save_path, numpy_data)


def tensor2im(input_image, imtype=np.uint8):
    """"
        tensor -> numpy , and normalize
    Parameters:
        input_image (tensor) []
        imtype (type)
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):  # reverse normalize
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        # (BGR)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # (channels, height, width) to (height, width, channels)

        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)
