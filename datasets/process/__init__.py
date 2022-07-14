#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/26
    Description:
        pre-process or post-process in pose
"""
from .affine_transform import get_affine_transform, exec_affine_transform, dark_get_affine_transform

from .pose_process import fliplr_joints, half_body_transform

from .heatmaps_process import get_max_preds, get_final_preds, generate_heatmaps, dark_get_final_preds

from .structure import *

from .coordinate_process import get_final_preds_coord