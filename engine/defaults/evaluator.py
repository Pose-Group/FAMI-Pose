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
import torch.nn
from tensorboardX import SummaryWriter

from datasets import build_eval_loader
from engine.core import build_core_function
from engine.defaults import VAL_PHASE, TEST_PHASE
from posetimation.zoo import build_model
from .base import BaseExecutor
from .checkpoints import get_all_checkpoints, get_latest_checkpoint


class DefaultEvaluator(BaseExecutor):

    def exec(self):
        self.eval()

    def __init__(self, cfg, output_folders: dict, phase=TEST_PHASE, **kwargs):
        super().__init__(cfg, output_folders, phase, **kwargs)

        cfg = self.cfg
        self.phase = phase
        self.dataloader = build_eval_loader(cfg, phase)
        self.model = build_model(cfg, phase=phase)
        self.dataset = self.dataloader.dataset
        self.GPUS = cfg.GPUS

        self.output = cfg.OUTPUT_DIR

        self.eval_from_checkpoint_id = kwargs.get("eval_from_checkpoint_id", -1)
        self.evaluate_model_state_files = []
        self.list_evaluate_model_files(cfg, phase)
        self.core_function = build_core_function(cfg, phase=phase, **kwargs)

        self.tb_writer_dict = {"writer": SummaryWriter(self.tb_save_folder),
                               "global_steps": 0}

    def list_evaluate_model_files(self, cfg, phase):
        subCfgNode = cfg.VAL if phase == VAL_PHASE else cfg.TEST
        if subCfgNode.MODEL_FILE:
            self.evaluate_model_state_files.append(subCfgNode.MODEL_FILE)
        else:
            if self.eval_from_checkpoint_id == -1:
                model_state_file = get_latest_checkpoint(self.checkpoints_save_folder)
                self.evaluate_model_state_files.append(model_state_file)
            else:
                candidate_model_files = get_all_checkpoints(self.checkpoints_save_folder)
                for model_file in candidate_model_files:
                    model_file_epoch_num = int(osp.basename(model_file).split("_")[1])
                    if model_file_epoch_num >= self.eval_from_checkpoint_id:
                        self.evaluate_model_state_files.append(model_file)

    def eval(self):
        for model_checkpoint_file in self.evaluate_model_state_files:
            model, epoch = self.model_load(model_checkpoint_file)
            # ############################################################
            # '''
            # HRNet 参数替换 - 用DarkPose训练好的HRNEt参数测试一下
            # '''
            # debug = False
            # if debug:
            #     hrent_file = '/media/Z/frunyang/FAMI-Pose/w48_384×288.pth'
            #     dark_hrnet = torch.load(hrent_file)
            #     model.hrnet.load_state_dict(dark_hrnet, True)
            ############################################################
            self.core_function.eval(model=model, dataloader=self.dataloader, tb_writer_dict=self.tb_writer_dict, epoch=epoch,
                                    phase=self.phase)

    def model_load(self, checkpoints_file):
        logger = logging.getLogger(__name__)
        logger.info("=> loading checkpoints from {}".format(checkpoints_file))
        if checkpoints_file == "skip":
            model = self.model
            if len(self.GPUS) > 1:
                model = torch.nn.DataParallel(model.cuda())
            else:
                model = model.cuda()
            return model, 0
        checkpoint_dict = torch.load(checkpoints_file)
        epoch = checkpoint_dict.get("begin_epoch", 0)
        # epoch = checkpoint_dict['begin_epoch']

        model = self.model

        # model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
        # model_state_dict = {}
        if "state_dict" in checkpoint_dict:
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
        else:
            model_state_dict = checkpoint_dict
        model_state_dict = {k.replace('preact.', '') if k.find('preact') == 0 else k: v for k, v in model_state_dict.items()}
        # model_state_dict = {k.replace('rough_pose_estimation_net.', ''): v for k, v in model_state_dict.items()}
        # model.load_state_dict(model_state_dict,strict=False)

        model.load_state_dict(model_state_dict)
        if len(self.GPUS) > 1:
            model = torch.nn.DataParallel(model.cuda())
        else:
            model = model.cuda()
        return model, epoch
