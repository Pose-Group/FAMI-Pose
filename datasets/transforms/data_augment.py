# -*-coding:utf-8-*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2021/5/5
    Description: 
"""
import cv2
from matplotlib import pyplot as plt

KEYPOINT_COLOR = (0, 255, 0)  # Green


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    # save_image('./test/a.jpg', image)
    # plt.imshow(image)

    return image
