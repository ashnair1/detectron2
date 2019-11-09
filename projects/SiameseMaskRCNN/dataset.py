import os
import random

import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.utils.visualizer import Visualizer


def get_xview2_dicts(data_dir, split):
    ann_path = os.path.join(data_dir, 'annotations', split +'.json')
    img_dir = os.path.join(data_dir, 'images', split)
    ann = load_coco_json(ann_path,
                         img_dir,
                         dataset_name='xView2_' + split)
    return ann


if __name__ == "__main__":
    if 'xView2_train' not in DatasetCatalog._REGISTERED.keys() or 'xView2_val' not in DatasetCatalog._REGISTERED.keys():
        # Load dataset
        data_dir = "./data/xview2/"
        for d in ["train", "val"]:
            DatasetCatalog.register("xView2_" + d, lambda d=d: get_xview2_dicts(data_dir, d))
            MetadataCatalog.get("xView2_" + d).set(thing_classes=["no-damage",
                                                                  "minor-damage",
                                                                  "major-damage",
                                                                  "destroyed"])

    #xview2_data = DatasetCatalog.get("xView2_train")
    xview2_metadata = MetadataCatalog.get("xView2_train")

    # Visualise
    dataset_dicts = DatasetCatalog.get("xView2_train")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=xview2_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        import matplotlib.pyplot as plt
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()
        #cv2.imshow('', vis.get_image()[:, :, ::-1])