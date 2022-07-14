#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import logging

import torch.optim as optimizer_zoo

logger = logging.getLogger(__name__)


def build_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.LR_SECOND_GROUP[0] is None:
        params_groups = filter(lambda p: p.requires_grad, model.parameters())
        params_groups = [
            {'params': params_groups, 'initial_lr': cfg.TRAIN.LR}
        ]
        logger.info(f"Default learning rate {cfg.TRAIN.LR}")
    else:
        ## 不同层不同学习率
        extra_param_ids = []
        extra_param_layers = cfg.TRAIN.LR_SECOND_GROUP

        for name, parameters in model.named_parameters():
            extra_ids = []
            for extra_layer_name in extra_param_layers:
                if name.startswith(extra_layer_name):
                    extra_ids.append(id(parameters))
            extra_param_ids.extend(extra_ids)

        # for layer_name in extra_param_layers:
        #     if layer_name == 'hrnet_old_param':
        #         extra_ids = []
        #         for name, parameters in model.named_parameters():
        #             if name.
        #             # if name in model.hrnet_old_param_name:
        #                 extra_ids.append(id(parameters))
        #     else:
        #         extra_ids = list(map(id, eval(f"model.{layer_name}.parameters()")))
        #     extra_param_ids.extend(extra_ids)

        base_params_group = filter(lambda p: p.requires_grad and id(p) not in extra_param_ids, model.parameters())
        extra_param_group = filter(lambda p: p.requires_grad and id(p) in extra_param_ids, model.parameters())

        params_groups = [
            [{"params": base_params_group}],
            [{"params": extra_param_group}]
        ]
        lr_list = [cfg.TRAIN.LR, cfg.TRAIN.LR_SECOND_GROUP_VALUE]
        logger.info(f"Default learning rate {cfg.TRAIN.LR}")
        logger.info(f"Second Params Group contain {extra_param_layers}. learning rate {cfg.TRAIN.LR_SECOND_GROUP_VALUE}")

    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optimizer_zoo.SGD(params_groups,
                                      lr=cfg.TRAIN.LR,
                                      momentum=cfg.TRAIN.MOMENTUM,
                                      weight_decay=cfg.TRAIN.WD,
                                      nesterov=cfg.TRAIN.NESTEROV
                                      )

    elif cfg.TRAIN.OPTIMIZER == 'adam':
        if len(params_groups) == 1:
            optimizer = optimizer_zoo.Adam(params_groups,lr=cfg.TRAIN.LR)
        elif len(params_groups) == 2:
            optimizer = []
            for i, pg in enumerate(params_groups):
                optimizer.append(optimizer_zoo.Adam(pg, lr=lr_list[i]))

    # optimizer = optimizer_zoo.Adam(params_groups, lr=cfg.TRAIN.LR)

    return optimizer
