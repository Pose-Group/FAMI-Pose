#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import os

from .my_custom import CfgNode


def update_config(cfg: CfgNode, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.rootDir:
        cfg.ROOT_DIR = args.rootDir

    cfg.OUTPUT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.OUTPUT_DIR))

    cfg.DATASET.JSON_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.JSON_DIR))
    cfg.DATASET.IMG_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.IMG_DIR))
    cfg.DATASET.TEST_IMG_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.TEST_IMG_DIR))

    if len(cfg.MODEL.PRETRAINED) > 0:
        cfg.MODEL.PRETRAINED = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.MODEL.PRETRAINED))
    cfg.MODEL.BACKBONE_PRETRAINED = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.MODEL.BACKBONE_PRETRAINED))

    cfg.VAL.ANNOT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.VAL.ANNOT_DIR))
    cfg.VAL.COCO_BBOX_FILE = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.VAL.COCO_BBOX_FILE))

    cfg.TEST.ANNOT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.TEST.ANNOT_DIR))
    cfg.TEST.COCO_BBOX_FILE = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.TEST.COCO_BBOX_FILE))

    cfg.freeze()


def get_cfg(args) -> CfgNode:
    """
        Get a copy of the default config.
        Returns:
            a fastreid CfgNode instance.
    """
    from .defaults import _C
    from .mppe_config import _C as MPPE_C

    if args.use_mppe_config:
        return MPPE_C.clone()
    else:
        return _C.clone()
