#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import logging

from .integral_loss import IntegralMSELoss, IntegralL1Loss
from .mse_loss import JointMSELoss

logger = logging.getLogger(__name__)


def build_loss(cfg, **kwargs):
    if "NAME" in cfg.LOSS:
        logger.warning("NAME 将会在之后被删除，请使用NAMES")
        if cfg.LOSS.NAME == "MSELOSS":
            return JointMSELoss(cfg.LOSS.USE_TARGET_WEIGHT)
        elif cfg.LOSS.NAME == "IntegralMSELoss":
            return IntegralMSELoss(True)
        elif cfg.LOSS.NAME == "IntegralL1Loss":
            return IntegralL1Loss(True)
