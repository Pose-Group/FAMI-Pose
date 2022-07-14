#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/10/08
    Description:
"""
import torch
import cv2
import numpy as np
from datasets.transforms.build import mean, std


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):  # 反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255  # 反ToTensor(),从[0,1]转为[0,255]
        # (BGR)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)

        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)
