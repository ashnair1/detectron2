import torch

from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler

__all__ = ["AdaptiveROIPooler"]


def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class AdaptiveROIPooler(ROIPooler):
    """
    The Adaptive Feature Pooler takes each proposal and maps them to all feature levels. This
    differs from the standard ROIPooler where each proposal is assigned to a particular
    feature level based on an equation (Refer Equation 1 in FPN paper).
    """

    def forward(self, x, box_lists):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (4*M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images over 4 feature levels. Every M boxes
                corresponds to a particular feature level in the pyramid. First M -> N5 and so on.
                C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes * num_level_assignments, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = list(range(level * num_boxes, (level + 1) * num_boxes))
            pooler_fmt_boxes_level = pooler_fmt_boxes
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)

        return output
