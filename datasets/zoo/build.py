#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/26
    Description:
"""
import torch.utils.data
# from datasets import DATASET_REGISTRY
import torch.utils.data.distributed

from engine.defaults.constant import DATASET_REGISTRY

__all__ = ["get_dataset_name", "build_train_loader", "build_eval_loader"]


def get_dataset_name(cfg):
    dataset_name = cfg.DATASET.NAME
    if dataset_name.startswith("PoseTrack"):
        dataset_version = "18" if cfg.DATASET.IS_POSETRACK18 else "17"
        dataset_name = dataset_name + dataset_version
    elif dataset_name == 'Jhmdb' or dataset_name == "Jhmdb2":
        dataset_version = cfg.DATASET.SPLIT_VERSION
        dataset_name = f"{dataset_name}_{dataset_version}"

    return dataset_name


# TODO Change to dataloader distributed in the future
# for train loader
def build_train_loader(cfg, **kwargs):
    cfg = cfg.clone()

    # for dataset_name in cfg.DATASETS.NAMES:
    # dataset_name = cfg.DATASET.M
    dataset_name = cfg.DATASET.NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase='train')

    batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return train_loader


# for validation / test loader
def build_eval_loader(cfg, phase):
    cfg = cfg.clone()

    # dataset_name = cfg.DATASET.NAME
    dataset_name = cfg.DATASET.NAME
    # dataset_name = cfg.DATASET.DATASET
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase=phase)
    if phase == 'validate':
        batch_size = cfg.VAL.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    elif phase == 'test':
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    else:
        raise BaseException

    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return eval_loader
