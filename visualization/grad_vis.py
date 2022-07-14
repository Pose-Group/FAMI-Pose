# -*-coding:utf-8-*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2021/4/24
    Description:
         Functions for gradient visualization
"""
import numpy as np
from utils.utils_image import format_np_output

def get_gradient_color_image(gradient):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = format_np_output(gradient)
    return gradient


def get_gradient_gray_image(gradient):
    """
        Converts 3d image to grayscale
    Args:
        gradient (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(gradient), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)

    grayscale_im = format_np_output(grayscale_im)

    return grayscale_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())

    pos_saliency, neg_saliency = format_np_output(pos_saliency), format_np_output(neg_saliency)
    return pos_saliency, neg_saliency


