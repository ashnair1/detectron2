# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN

def add_panet_config(cfg):
    """
    Add config for PANet.
    """

    # Fully Connected Fusion
    cfg.MODEL.ROI_MASK_HEAD.FCF = CN()
    cfg.MODEL.ROI_MASK_HEAD.FCF.ENABLED = True
    cfg.MODEL.ROI_MASK_HEAD.FCF.NUM_CONV_FC = 2
