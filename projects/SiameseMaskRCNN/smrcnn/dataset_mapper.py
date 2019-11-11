import copy
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
        return dataset_dict
