import os
import cv2
import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer

import sys
sys.path.append("/home/an1/detectron2/projects/PANet/")

import dataset
from dataset import DatasetMapper, DotaMapper

def output(vis, fname, show):
    if show:
        print(fname)
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.imshow("window", vis.get_image()[:, :, ::-1])
        cv2.waitKey()
    else:
        vis_dir = "/home/an1/detectron2/projects/PANet/"
        filepath = os.path.join(vis_dir, "results", fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)


if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file("/home/an1/detectron2/projects/PANet/configs/dota_obb/faster_rcnn_R_50_FPN_1x_rotated.yaml")
    cfg.freeze()
    train_data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=DotaMapper(cfg, False))
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    end_flag = 0

    for batch in train_data_loader:
        if not end_flag:
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0)
                if cfg.INPUT.FORMAT == "BGR":
                    img = img[:, :, [2, 1, 0]]
                else:
                    img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

                visualizer = Visualizer(img, metadata=metadata, scale=2.0)
                assert "annotations" in list(per_image.keys())
                vis = visualizer.draw_dataset_dict(per_image)
                output(vis, str(per_image["image_id"]) + ".jpg", False)
                