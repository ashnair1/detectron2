from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

import copy
import logging
import numpy as np
import torch


class DotaMapper(DatasetMapper):
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
        # Only HFlip and Resize are supported for rotated_boxes
        #augs = [T.RandomFlip(0.4, horizontal=True, vertical=False)]

        if self.is_train:
            tfm_gens = self.tfm_gens #+ augs
        else:
            tfm_gens = self.tfm_gens

        logging.getLogger(__name__).info("Original Augmentation: " + str(self.tfm_gens))

        logging.getLogger(__name__).info("Updated Augmentation List: " + str(tfm_gens))

        image, transforms = T.apply_transform_gens(tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        for a in dataset_dict['annotations']:
            a['bbox'] = transforms.apply_rotated_box(np.asarray([a['bbox']]))[0].tolist()

        annos = dataset_dict['annotations']
        instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict