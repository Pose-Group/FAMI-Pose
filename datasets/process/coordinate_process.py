import numpy as np
import math
from .heatmaps_process import transform_preds


def get_final_preds_coord(batch_keypoints, center, scale, heatmap_width, heatmap_height):
    # batch_keypoints = batch_keypoints.reshape(batch_keypoints.shape[0], 17, 3)
    coords, maxvals = batch_keypoints[:, :, 0:2], batch_keypoints[:, :, 2:3]

    # 反归一化
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * heatmap_width  # x
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * heatmap_height  # y

    preds = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):  # batch
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals
