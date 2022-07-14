#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import logging
import os.path as osp

import torch

from utils.utils_folder import list_immediate_childfile_paths


def get_latest_checkpoint(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None

    latest_checkpoint = checkpoint_saves_paths[0]
    # we define the format of checkpoint like "epoch_0_state.pth"
    latest_index = int(osp.basename(latest_checkpoint).split("_")[1])
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoint_save_file_name = osp.basename(checkpoint_save_path)
        now_index = int(checkpoint_save_file_name.split("_")[1])
        if now_index > latest_index:
            latest_checkpoint = checkpoint_save_path
            latest_index = now_index
    return latest_checkpoint


def get_all_checkpoints(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None
    checkpoints_list = []
    # we define the format of checkpoint like "epoch_0_state.pth"
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoints_list.append(checkpoint_save_path)
    return checkpoints_list


def save_checkpoint(epoch, save_folder, model, optimizer, **kwargs):
    model_save_path = osp.join(save_folder, 'epoch_{}_state.pth'.format(epoch))
    checkpoint_dict = dict()

    # Because nn.DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/5
    model_state_dict = model.state_dict()
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

    checkpoint_dict['begin_epoch'] = epoch
    checkpoint_dict['state_dict'] = model_state_dict

    optimizer_state_dict = []
    if isinstance(optimizer, list):
        for op in optimizer:
            optimizer_state_dict.append(op.state_dict())
    else:
        optimizer_state_dict.append(optimizer.state_dict())
    checkpoint_dict['optimizer'] = optimizer_state_dict
    torch.save(checkpoint_dict, model_save_path)

    return model_save_path


def resume(model, optimizer, checkpoint_file, **kwargs):
    checkpoint = torch.load(checkpoint_file)
    begin_epoch = checkpoint['begin_epoch'] + 1
    gpus = kwargs.get("gpus", [])
    if len(gpus) <= 1:
        state_dict = {k.replace('module.', '') if k.find('module') == 0 else k: v for k, v in checkpoint['state_dict'].items()}
        # state_dict = {k.replace('module.', '') if k.index('module') == 0 else k: v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint["state_dict"]

    state_dict = {k.replace('preact.', '') if k.find('preact') == 0 else k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    optimizer_state_dict = checkpoint['optimizer']
    if isinstance(optimizer, list):
        assert type(optimizer_state_dict) == type(optimizer)
        for opz in optimizer:
            for op_sd in optimizer_state_dict:
                if len(opz.state_dict()['param_groups'][0]['params']) == len(op_sd['param_groups'][0]['params']):
                    opz.load_state_dict(op_sd)
                    for state in opz.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
                    optimizer_state_dict.remove(op_sd)
                    break
                else:
                    logger = logging.getLogger(__name__)
                    logger.error("bad resume")
    else:
        optimizer.load_state_dict(optimizer_state_dict[0])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    return model, optimizer, begin_epoch
