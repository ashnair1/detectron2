import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json # How we get dataset_dicts


def register_isaid():
    # Register iSAID Dataset (in COCO format)
    data_dir = "/home/an1/detectron2/datasets/isaid"

    for d in ["train", "val"]:
        DatasetCatalog.register("isaid_" + d, lambda d=d: load_coco_json(json_file=data_dir + '/annotations/instances_{}.json'.format(d),
                                                                        image_root=data_dir + '/images/{}/'.format(d),
                                                                        dataset_name="isaid_{}".format(d)))
        MetadataCatalog.get("isaid_" + d).json_file = os.path.join(data_dir, "annotations",
                                                                   "instances_{}.json".format(d))