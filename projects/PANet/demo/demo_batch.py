# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import torch

import panet
from panet import add_panet_config

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_panet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--input_dir", help="Directory of images to be tested")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--weights",
        help="Path to weights",
        default=[]
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def load_img(img_path, transform_gen):
    img1 = read_image(img_path, format="BGR")
    img = transform_gen.get_transform(img1).apply_image(img1)
    img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    return img1, img_tensor

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    model = build_model(cfg) # returns a torch.nn.Module

    transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    DetectionCheckpointer(model).load(args.weights) # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
    model.train(False) # inference mode

    imgs = []
    inputs = []
    im_paths = []
    for im in os.listdir(args.input_dir):
        img = os.path.join(args.input_dir, im)
        img1, img_tensor = load_img(img, transform_gen)
        im_paths.append(img)
        imgs.append(img1)
        inputs.append({"image":img_tensor, "height": img1.shape[0], "width": img1.shape[1]})
    
    # Batch inference
    outputs = model(inputs)
    
    metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

    for res, im, path in zip(outputs, imgs, im_paths):
        im = im[:, :, ::-1]
        visualizer = Visualizer(im, metadata, instance_mode=ColorMode.IMAGE)
        instances = res["instances"].to(torch.device("cpu"))
        # Detach box tensor to convert it into numpy arrays in visualizer
        instances.pred_boxes.tensor = instances.pred_boxes.tensor.detach()
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            vis_output.save(out_filename)
