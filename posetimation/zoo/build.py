#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
from engine.defaults import TRAIN_PHASE
from engine.defaults.constant import MODEL_REGISTRY

def build_model(cfg, phase, **kwargs):
    """
        return : model Instance
    """
    model_name = cfg.MODEL.NAME

    if model_name == "SimpleBaseline":
        model_Class = MODEL_REGISTRY.get(model_name)
        model_instance = model_Class.get_net(cfg=cfg, phase=phase, **kwargs)
    elif model_name == 'VisTR':
        args = cfg
        num_classes = 17

        aux_loss = False
        args.defrost()
        args.backbone = "resnet101"
        args.dec_layers = 6
        args.dim_feedforward = 2048
        args.dropout = 0.1
        args.enc_layers = 6
        args.hidden_dim = 384
        args.nheads = 8
        args.position_embedding = 'sine'
        args.lr_backbone = 1e-05
        args.masks = True
        args.dilation = False
        args.pre_norm = False
        args.num_frames = 3
        args.num_queries = args.num_frames * 17

        from posetimation.zoo._discarded.Transformer import build_backbone
        from posetimation.zoo._discarded.Transformer.vistr.transformer import build_transformer
        from posetimation.zoo._discarded.Transformer import VisTR, VisTRsegm
        backbone = build_backbone(args)

        transformer = build_transformer(args)

        model = VisTR(
            backbone,
            transformer,
            num_classes=num_classes,  # keypoints
            num_frames=args.num_frames,
            num_queries=args.num_queries,
            aux_loss=aux_loss,
        )
        # if args.masks:
        model_instance = VisTRsegm(model)

        # model_Class = MODEL_REGISTRY.get(model_name)
        # model_instance = model_Class.build(cfg)

        return model_instance
    else:
        model_instance = MODEL_REGISTRY.get(model_name)(cfg, phase, **kwargs)

    if phase == TRAIN_PHASE and cfg.MODEL.INIT_WEIGHTS:
        # model_instance.init_weights()
        model_instance.train()

    if phase != TRAIN_PHASE:
        model_instance.eval()

    return model_instance


def get_model_hyperparameter(cfg, **kwargs):
    """
        return : model class
    """
    model_name = cfg.MODEL.NAME

    if model_name == "VisTR":
        return "VisTransformer"
    else:
        hyper_parameters_setting = MODEL_REGISTRY.get(model_name).get_model_hyper_parameters(cfg)

    return hyper_parameters_setting
