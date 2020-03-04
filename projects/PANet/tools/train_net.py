"""
PANet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

import sys
sys.path.append("/home/an1/detectron2/projects/PANet/")

from panet import add_panet_config
from dataset import data_dict


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        data = data_dict[cfg.DATASETS.TEST[0].split('_')[0]]
        data_evaluator = data['evaluator'] if 'evaluator' in list(data.keys()) else COCOEvaluator
        return data_evaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        data = data_dict[cfg.DATASETS.TEST[0].split('_')[0]]
        data_mapper = data['mapper'] if 'mapper' in list(data.keys()) else DatasetMapper
        return build_detection_test_loader(cfg, dataset_name, mapper=data_mapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        data = data_dict[cfg.DATASETS.TRAIN[0].split('_')[0]]
        data_mapper = data['mapper'] if 'mapper' in list(data.keys()) else DatasetMapper
        return build_detection_train_loader(cfg, mapper=data_mapper(cfg, True))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
