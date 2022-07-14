#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import argparse
import os.path as osp


def default_parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', type=str, required=True)
    parser.add_argument('--PE_Name', type=str, default='DcPose')
    parser.add_argument('-dn', '--detector_name', type=str, default='DcPose')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--val_from_checkpoint',
                        help='exec val from the checkpoint_id. if config.yaml specifies a model file, this parameter if invalid',
                        type=int,
                        default='-1')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--root_dir', type=str, default='../')
    parser.add_argument('--use_mppe_config', action='store_true', default=True)
    parser.add_argument('--dis_mppe_config', action='store_true', default=False)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.dis_mppe_config:
        args.use_mppe_config = False
    args.rootDir = osp.abspath(args.root_dir)
    # args.cfg = osp.join(osp.abspath(args.cfg))
    args.cfg = osp.join(args.rootDir, osp.abspath(args.cfg))

    args.PE_Name = args.PE_Name.upper()
    return args
