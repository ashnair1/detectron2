from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json # How we get dataset_dicts

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

import numpy as np
import cv2
import random
import copy
import torch

# __all__ = [
#     "register_isaid",
#     #"isaid_mapper"
# ]

def register_isaid():
    # Register iSAID Dataset (in COCO format)
    data_dir = "/home/an1/detectron2/datasets/isaid"

    for d in ["train", "val"]:
        DatasetCatalog.register("isaid_" + d, lambda d=d: load_coco_json(json_file=data_dir + '/annotations/instances_{}.json'.format(d),
                                                                        image_root=data_dir + '/images/{}/'.format(d),
                                                                        dataset_name="isaid_{}".format(d)))


# def isaid_mapper(dataset_dict):
#     # Implement a mapper, similar to the default DatasetMapper, but with own customizations
#     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
#     # Custom augs
#     tfm_gens = [T.RandomFlip(0.5, horizontal=True),
#                 T.RandomFlip(0.6, horizontal=False, vertical=True)]
    
#     image, transforms = T.apply_transform_gens(tfm_gens, image)
#     dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dict.pop("annotations")
#         if obj.get("iscrowd", 0) == 0
#     ]
#     instances = utils.annotations_to_instances(annos, image.shape[:2])
#     dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict