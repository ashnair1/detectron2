# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import sys
import onnx

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import add_export_config, export_caffe2_model, export_onnx_model
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

sys.path.append("/home/an1/detectron2/projects/PANet/")
from panet import add_panet_config


def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    add_panet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model to Caffe2 or Onnx")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--export-format", default="caffe2", help="format for model to exported in")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

    # get a sample data
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    first_batch = next(iter(data_loader))

    assert args.export_format in ["caffe2", "onnx"], "Unexpected export format:{}".format(args.export_format)

    # convert and save model
    if args.export_format == "caffe2":
        caffe2_model = export_caffe2_model(cfg, torch_model, first_batch)
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=first_batch)
        model = caffe2_model
    else:
        if args.run_eval:
            logger.info("Evaluation only supported for caffe2 models")
            args.run_eval = False
        onnx_model = export_onnx_model(cfg, torch_model, first_batch)
        # save the onnx model
        onnx.save(onnx_model, os.path.join(args.output, "model.onnx"))
        model = onnx_model

    # run evaluation with the converted model
    if args.run_eval:
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, cfg, True, args.output)
        metrics = inference_on_dataset(model, data_loader, evaluator)
        print_csv_format(metrics)
