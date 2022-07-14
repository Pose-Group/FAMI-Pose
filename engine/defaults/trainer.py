#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import logging

import torch
import torch.nn
from tensorboardX import SummaryWriter

from datasets import build_train_loader
from engine.core import build_core_function
from engine.defaults import TRAIN_PHASE
from posetimation.loss import build_loss
from posetimation.optimizer import build_lr_scheduler, build_optimizer
from posetimation.zoo import build_model
from .base import BaseExecutor
from .checkpoints import get_latest_checkpoint, save_checkpoint, resume


class DefaultTrainer(BaseExecutor):
    def exec(self):
        self.train()

    def __init__(self, cfg, output_folders: dict, **kwargs):
        super().__init__(cfg, output_folders, TRAIN_PHASE, **kwargs)

        # cfg = self.cfg
        self.dataloader = build_train_loader(cfg)
        self.model = build_model(cfg, phase='train')
        self.optimizer = build_optimizer(cfg, self.model)
        self.lr_scheduler = None

        # self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
        self.loss_criterion = build_loss(cfg)
        self.begin_epoch = 0
        self.end_epoch = cfg.TRAIN.END_EPOCH
        self.save_model_per_epoch = cfg.TRAIN.SAVE_MODEL_PER_EPOCH
        self.GPUS = cfg.GPUS
        self.model = self.model.cuda()
        if self.lr_scheduler is None:
            self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)

        self.model_resume()
        self.core_function = build_core_function(cfg, criterion=self.loss_criterion, **kwargs)

        self.tb_writer_dict = {"writer": SummaryWriter(self.tb_save_folder),
                               "global_steps": 0}

    def train(self):
        logger = logging.getLogger(__name__)
        # self.model_resume()
        if len(self.GPUS) > 1:
            self.model = torch.nn.DataParallel(self.model)
        logger.info(f"=> end_epoch:{self.end_epoch} LR_Step:{','.join([str(i) for i in self.cfg.TRAIN.LR_STEP])} LR_FACTOR:{self.cfg.TRAIN.LR_FACTOR}")

        for epoch in range(self.begin_epoch, self.end_epoch):
            last_lr = []
            if isinstance(self.lr_scheduler, list):
                for lr_s in self.lr_scheduler:
                    last_lr.append(lr_s.get_last_lr())
            else:
                last_lr.append(self.lr_scheduler.get_last_lr())

            logger.info('=> Start train epoch {}, last_lr{}'.format(epoch, last_lr))
            self.core_function.train(model=self.model, epoch=epoch, optimizer=self.optimizer, dataloader=self.dataloader,
                                     tb_writer_dict=self.tb_writer_dict)
            if isinstance(self.lr_scheduler, list):
                for lr_s in self.lr_scheduler:
                    lr_s.step()
            else:
                self.lr_scheduler.step()
            # self.lr_scheduler.step()
            if epoch % self.save_model_per_epoch == 0:
                model_save_path = self.save_model(epoch)
                logger.info('=> Saved epoch {} model state to {}'.format(epoch, model_save_path))

            # record learning_rate
            writer = self.tb_writer_dict["writer"]

            if isinstance(self.lr_scheduler, list):
                for index, lr_s in enumerate(self.lr_scheduler):
                    writer.add_scalar(f'lr_scheduler_{index}', last_lr[index], epoch)
            else:
                writer.add_scalar('lr_scheduler', last_lr, epoch)

    def save_model(self, epoch):
        model_save_path = save_checkpoint(epoch, self.checkpoints_save_folder, self.model, self.optimizer)
        return model_save_path

    def model_resume(self):
        logger = logging.getLogger(__name__)
        checkpoint_file = get_latest_checkpoint(self.checkpoints_save_folder)
        if checkpoint_file is not None:
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            # self.model, self.begin_epoch = resume(self.model, checkpoint_file, gpus=self.GPUS)
            self.model, self.optimizer, self.begin_epoch = resume(self.model, self.optimizer, checkpoint_file, gpus=self.GPUS)
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer, last_epoch=self.begin_epoch)
            # logger.info(f"=> last lr {self.lr_scheduler.get_last_lr()}")
        else:
            logger.warning("=> no checkpoint file available to resume")

    def __del__(self):
        super(DefaultTrainer, self).__del__()
