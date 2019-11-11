import os
import random

import cv2

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from dataset_mapper import SiameseDataMapper

def get_xview2_dicts(data_dir, split):
    ann_path = os.path.join(data_dir, 'annotations', split +'.json')
    img_dir = os.path.join(data_dir, 'images', split)
    ann = load_coco_json(ann_path,
                         img_dir,
                         dataset_name='xView2_' + split)
    for a in ann:
        img_name = '_'.join(['post' if i == 'pre' else i for i in a['file_name'].split('_')])
        img_name = os.path.basename(img_name)
        a['post_file_name'] = os.path.join(data_dir, 'images', split + '_post', img_name)
        a['pre_file_name'] = a['file_name']
        del(a['file_name'])
    return ann


if __name__ == "__main__":
    if 'xView2_train' not in DatasetCatalog._REGISTERED.keys() or 'xView2_val' not in DatasetCatalog._REGISTERED.keys():
        # Load dataset
        data_dir = "/media/ashwin/DATA/detectron2/projects/SiameseMaskRCNN/data/xview2/"
        for d in ["train", "val"]:
            DatasetCatalog.register("xView2_" + d, lambda d=d: get_xview2_dicts(data_dir, d))
            MetadataCatalog.get("xView2_" + d).set(thing_classes=["no-damage",
                                                                  "minor-damage",
                                                                  "major-damage",
                                                                  "destroyed"])


    # Visualise
    # dataset_dicts = DatasetCatalog.get("xView2_train")
    # xview2_metadata = MetadataCatalog.get("xView2_train")
    #
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["post_file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=xview2_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     import matplotlib.pyplot as plt
    #     plt.imshow(vis.get_image()[:, :, ::-1])
    #     plt.show()
    #     #cv2.imshow('', vis.get_image()[:, :, ::-1])

    # Test Data Loader
    cfg = get_cfg()
    # add_tridentnet_config(cfg)
    cfg.merge_from_file('/media/ashwin/DATA/detectron2/projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml')
    cfg.freeze()
    tdl = build_detection_test_loader(cfg, "xView2_val", mapper=SiameseDataMapper(cfg, False))