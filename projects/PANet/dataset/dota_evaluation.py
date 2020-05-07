import contextlib
import copy
import datetime
import io
import itertools
import json
import logging
import math
import numpy as np
import os
import shutil
import torch
from collections import OrderedDict, defaultdict
import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import RotatedCOCOEvaluator
from detectron2.structures import Boxes, BoxMode
from fvcore.common.file_io import PathManager, file_lock
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)

class DOTAEvaluator(RotatedCOCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._logger.info("Converting {} to COCO format".format(dataset_name))

        # Save results in DOTA format
        self.save_dota = True

        cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
        convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(cache_path)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        assert self._do_evaluation is True

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """

        self._dota_preds = defaultdict(list)

        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            prediction = {"image_id": image_id}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)

                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()

                for box, score, cls in zip(boxes, scores, classes):
                    quad = xywha_to_quad(box)
                    # Flatten list of tuples 
                    quad = list(sum(quad, ()))
                    assert len(quad) == 8, "Invalid quadrilateral"
                    self._dota_preds[cls].append(
                        f"{image_id} {score:.3f} {quad[0]:.1f} {quad[1]:.1f} {quad[2]:.1f} {quad[3]:.1f}" \
                            f" {quad[4]:.1f} {quad[5]:.1f} {quad[6]:.1f} {quad[7]:.1f}"
                    )

                prediction["instances"] = self.instances_to_json(instances, input["image_id"])
                prediction["dota_instances"] = self._dota_preds
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        dota_det_dict = dict((k, []) for k in range(len(self._metadata.thing_classes)))
        for p in predictions:
            for k in p['dota_instances'].keys():
                dota_det_dict[k] += p['dota_instances'][k]
            p.pop('dota_instances')

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)
            
            if self.save_dota:
                dirpath = os.path.join(self._output_dir, "dota_dets")
                if PathManager.exists(dirpath) is False:
                    PathManager.mkdirs(dirpath)
                else:
                    shutil.rmtree(dirpath)
                    PathManager.mkdirs(dirpath)
                # Write class detections
                for cls_id, cls_name in enumerate(self._metadata.thing_classes):
                    dfilename = os.path.join(dirpath, f'Task1_{cls_name}.txt')
                    with open(dfilename, 'w') as j:
                        for p in dota_det_dict[cls_id]:
                            j.write("%s\n" % p)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            assert task == "bbox", "Task {} is not supported".format(task)
            coco_eval = (
                self._evaluate_predictions_on_coco(self._coco_api, coco_results)
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.info(f"Cached annotations in COCO format already exist: {output_file}")
        else:
            logger.info(f"Converting dataset annotations in '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            with PathManager.open(output_file, "w") as json_file:
                logger.info(f"Caching annotations in COCO format: {output_file}")
                json.dump(coco_dict, json_file)


def convert_to_coco_dict(dataset_name):
    """
    Convert a dataset in detectron2's standard format into COCO json format

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name:
            name of the source dataset
            must be registered in DatastCatalog and in detectron2's standard format
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]

            # TODO: BoxMode when stored in json is serialised so when read it gives int value
            bbox_mode = BoxMode.XYWHA_ABS
            #bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)
            #bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = (bbox[2] * bbox[3])
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)
            coco_annotation["category_id"] = reverse_id_mapper(annotation["category_id"])


            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict

def xywha_to_quad(bbox):
    """
    Return list of tuples of quadrilateral points
    """
    cnt_x, cnt_y, w, h, angle = bbox
    theta = angle * math.pi / 180.0
    c = math.cos(theta)
    s = math.sin(theta)
    rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
    # x: left->right ; y: top->down
    rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
    return rotated_rect