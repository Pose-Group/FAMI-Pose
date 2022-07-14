#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import torch
import torch.nn as nn


class JointMSELoss(nn.Module):
    def __init__(self, use_target_weight: bool = True, divided_num_joints=True):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.use_target_weight = use_target_weight
        self.divided_num_joints = divided_num_joints

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                # loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
                # loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        if self.divided_num_joints:
            loss = loss / num_joints

        return loss

