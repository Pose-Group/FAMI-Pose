#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import torch.nn as nn




class BaseModel(nn.Module):

    def forward(self, *input):
        raise NotImplementedError

    def init_weights(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_model_hyper_parameters(cls, cfg):
        raise NotImplementedError
