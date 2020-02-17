import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.mask_head import BaseMaskRCNNHead
from detectron2.utils.registry import Registry

from .adaptive_pooler import AdaptiveROIPooler

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")


@ROI_HEADS_REGISTRY.register()
class PANetROIHeads(StandardROIHeads):
    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = AdaptiveROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = AdaptiveROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHeadAdpp(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # Add additional fcs for each level
        num_levels = 4
        for i in range(num_levels - 1):
            self.fcs.insert(1, self.fcs[0])

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            x1 = list(torch.split(x, int(x.shape[0] // 4), dim=0))
            lvls = []

            # Pooled feature grids go through 1 parameter layer independently
            for layer, lvl_feat in zip(self.fcs[:-1], x1):
                lvls.append(F.relu(layer(lvl_feat)))

            # Fusion op - Max
            m = lvls[0]
            for i in lvls:
                m = torch.max(m, i)

            # Fc2
            x = F.relu(self.fcs[-1](m))
        return x

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHeadAdpp(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg, input_shape)

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        # Add additional convs for each level
        num_levels = 4
        for i in range(num_levels - 1):
            self.conv_norm_relus.insert(1, self.conv_norm_relus[0])

        # Fully connected fusion
        self.conv_fc_norm_relus = []
        if cfg.MODEL.ROI_MASK_HEAD.FCF.ENABLED:
            num_conv_fc = cfg.MODEL.ROI_MASK_HEAD.FCF.NUM_CONV_FC
            for k in range(num_conv_fc):
                conv_fc = Conv2d(
                    conv_dims,
                    int(conv_dims/2) if k == num_conv_fc - 1 else conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv_fc)
                self.add_module("mask_fcn_fc{}".format(k + 1), conv_fc)
                self.conv_fc_norm_relus.append(conv_fc)

            self.mask_fc = nn.Sequential(
                nn.Linear(int(conv_dims/2) * (cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION ** 2),
                          (2*cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION) ** 2),
                nn.ReLU(inplace=True))
            for m in self.mask_fc.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(m.weight)
                    nn.init.constant_(m.bias, 0.01)

            self.add_module("mask_fc", self.mask_fc)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def layers(self, x):
        x1 = list(torch.split(x, int(x.shape[0] // 4), dim=0))
        lvls = []

        # Pooled feature grids go through 1 parameter layer independently
        for layer, lvl_feat in zip(self.conv_norm_relus[:-3], x1):
            lvls.append(layer(lvl_feat))

        # Fusion op - Max
        x = lvls[0]
        for i in lvls:
            x = torch.max(x, i)

        for num, layer in enumerate(self.conv_norm_relus[4:]):
            x = layer(x)
            if num == len(self.conv_norm_relus[4:]) - 2:
                xb = x
        x = F.relu(self.deconv(x))
        x = self.predictor(x)

        num_classes = x.shape[1]
        pooler_resolution = int(x.shape[-1] / 2)

        # Fully connected fusion branch
        if self.conv_fc_norm_relus:
            for layer in self.conv_fc_norm_relus:
                xb = layer(xb)
            # Hack for batch size of zero
            if xb.shape[0]:
                xb = self.mask_fc(xb.view(xb.shape[0], -1))
            else:
                xb = self.mask_fc(xb.view(xb.shape[0], np.prod(xb.shape[1:])))
            xb = xb.view(xb.shape[0], 1, 2*pooler_resolution, 2*pooler_resolution)
            xb = xb.repeat(1, num_classes, 1, 1)
            x = xb + x

        return x


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
