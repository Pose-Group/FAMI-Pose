#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
TRAIN_PHASE = "train"
VAL_PHASE = "validate"
TEST_PHASE = "test"



from .argument_parser import default_parse_args
from .runner import DefaultRunner
from .trainer import DefaultTrainer
