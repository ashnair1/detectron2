# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN

def add_panet_config(cfg):
    """
    Add config for PANet.
    """
    # Filter annotations by area
    cfg.DATALOADER.FILTER_ANNOTATIONS_AREA = CN()
    cfg.DATALOADER.FILTER_ANNOTATIONS_AREA.MIN_AREA = 0
    cfg.DATALOADER.FILTER_ANNOTATIONS_AREA.MAX_AREA = 1e5

    # Fully Connected Fusion
    cfg.MODEL.ROI_MASK_HEAD.FCF = CN()
    cfg.MODEL.ROI_MASK_HEAD.FCF.ENABLED = True
    cfg.MODEL.ROI_MASK_HEAD.FCF.NUM_CONV_FC = 2
