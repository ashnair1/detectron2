from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

import copy
import logging
import torch


def filter_small_instances(instances, min, max):
    """
    Filter out instances with area outside area range [min, max) in an `Instances` object.

    Args:
        instances (Instances):

    Returns:
        Instances: the filtered instances.
    """
    p = instances.gt_boxes.tensor
    w = instances.gt_boxes.tensor[:, 2] - instances.gt_boxes.tensor[:, 0]
    h = instances.gt_boxes.tensor[:, 3] - instances.gt_boxes.tensor[:, 1]
    areas = w * h
    inds = ((areas < min) | (areas > max)).nonzero()

    mask = ((p[:, 3] - p[:, 1]) * (p[:, 2] - p[:, 0]) >= min) & ((p[:, 3] - p[:, 1]) * (p[:, 2] - p[:, 0]) < max)
    instances.gt_boxes.tensor = instances.gt_boxes.tensor[mask]
    instances.gt_classes      = instances.gt_classes[mask]

    assert p.shape[0] == inds.shape[0] + instances.gt_boxes.tensor.shape[0]

    #instances.gt_boxes.tensor = p[(p[:, 3] - p[:, 1]) * (p[:, 2] - p[:, 0]) > min]
    if instances.has("gt_masks") and inds.nelement():
        instances.gt_masks.polygons = [v for i, v in enumerate(instances.gt_masks.polygons) if i not in frozenset(inds.flatten().tolist())]

    assert (instances.gt_boxes.tensor.shape[0] == instances.gt_classes.shape[0] == 0) or (
                instances.gt_boxes.tensor.shape[0] == instances.gt_classes.shape[0] == len(instances.gt_masks.polygons))

    return instances


class ISAIDMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        self.min_area = cfg.DATALOADER.FILTER_ANNOTATIONS_AREA.MIN_AREA
        self.max_area = cfg.DATALOADER.FILTER_ANNOTATIONS_AREA.MAX_AREA

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # Implement a mapper, similar to the default DatasetMapper, but with own customizations
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        
        # Custom augs to be used while training
        augs = [T.RandomFlip(0.6, horizontal=False, vertical=True)]

        if self.is_train:
            tfm_gens = self.tfm_gens + augs
        else:
            tfm_gens = self.tfm_gens

        logging.getLogger(__name__).info("Original Augmentation: " + str(self.tfm_gens))

        logging.getLogger(__name__).info("Updated Augmentation List: " + str(tfm_gens))

        image, transforms = T.apply_transform_gens(tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        dataset_dict["instances"] = filter_small_instances(dataset_dict["instances"], self.min_area, self.max_area)
        return dataset_dict
