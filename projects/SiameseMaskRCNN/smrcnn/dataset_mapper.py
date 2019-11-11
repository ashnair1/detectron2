import copy
import numpy as np
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T


class SiameseDataMapper(DatasetMapper):
    """
    A customized version of `detectron2.data.DatasetMapper` for reading in two images.
    """
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=True)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        preim = utils.read_image(dataset_dict["pre_file_name"], format=self.img_format)
        postim = utils.read_image(dataset_dict["post_file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, preim)
        utils.check_image_size(dataset_dict, postim)

        dataset_dict["pre_image"] = torch.as_tensor(preim.transpose(2, 0, 1).astype("float32"))
        dataset_dict["post_image"] = torch.as_tensor(postim.transpose(2, 0, 1).astype("float32"))

        if "annotations" not in dataset_dict:
            preim, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, preim
            )
            postim, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, postim
            )
        else:
            # # Crop around an instance if there are instances in the image.
            # # USER: Remove if you don't use cropping
            # if self.crop_gen:
            #     crop_tfm = utils.gen_crop_transform_with_instance(
            #         self.crop_gen.get_crop_size(preim.shape[:2]),
            #         preim.shape[:2],
            #         np.random.choice(dataset_dict["annotations"]),
            #     )
            #     preim = crop_tfm.apply_image(preim)
            #     postim = crop_tfm.apply_image(postim)
            preim, transforms = T.apply_transform_gens(self.tfm_gens, preim)
            postim, transforms = T.apply_transform_gens(self.tfm_gens, postim)
            # if self.crop_gen:
            #     transforms = crop_tfm + transforms

        image_shape = preim.shape[:2]

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(
            #         obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            #     )
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            annos = dataset_dict["annotations"]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            # # Create a tight bounding box from masks, useful when image is cropped
            # if self.crop_gen and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
