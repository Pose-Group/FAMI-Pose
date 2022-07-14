#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import logging

import torch.optim


def build_lr_scheduler(cfg, optimizer, **kwargs):
    if cfg.TRAIN.LR_SCHEDULER == 'MultiStepLR':
        last_epoch = kwargs["last_epoch"] if 'last_epoch' in kwargs else -1

        if not isinstance(optimizer, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)
        else:
            lr_scheduler = []
            for op in optimizer:
                lr_scheduler.append(
                    torch.optim.lr_scheduler.MultiStepLR(op, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch))

        # for i in range(last_epoch):
        #     lr_scheduler.step()

        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.MILESTONES, cfg.TRAIN.GAMMA, last_epoch=last_epoch)
    else:
        logger = logging.getLogger(__name__)
        logger.error("Please Check if LR_SCHEDULER is valid")
        raise Exception("Please Check if LR_SCHEDULER is valid")

    return lr_scheduler
