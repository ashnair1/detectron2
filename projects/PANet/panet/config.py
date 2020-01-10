# -*- coding: utf-8 -*-


def add_panet_config(cfg):
    """
    Add config for PANet.
    """

    # Fully Connected Fusion
    cfg.MODEL.ROI_MASK_HEAD.FCF = True
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV_FC = 2
