import argparse

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, models
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.export.caffe2_inference import ProtobufDetectionModel
from detectron2.utils.visualizer import ColorMode, Visualizer

import sys
sys.path.insert(0,"/home/an1/detectron2")

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 caffe2 demo")

    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
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

    args = get_parser().parse_args()

    cfg = setup_cfg(args)

    IMAGE_LOCATION = "/home/an1/detectron2/caffe2_model/input.jpg"
    INIT_NET = '/home/an1/detectron2/caffe2_model/model_init.pb'
    PREDICT_NET = '/home/an1/detectron2/caffe2_model/model.pb'

    transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    img, img_tensor = load_img(IMAGE_LOCATION, transform_gen)

    device_opts = core.DeviceOption(caffe2_pb2.CPU)

    # Read the contents of the input protobufs into local variables
    init_net = caffe2_pb2.NetDef()
    with open(INIT_NET, 'rb') as f:
        init_net.ParseFromString(f.read())
        init_net.device_option.CopyFrom(device_opts)

    predict_net = caffe2_pb2.NetDef()
    with open(PREDICT_NET, "rb") as f:
        predict_net.ParseFromString(f.read())
        predict_net.device_option.CopyFrom(device_opts)

    c2det = ProtobufDetectionModel(predict_net, init_net)
    results = c2det([{"image":img_tensor, "height": img.shape[0], "width": img.shape[1]}])

    # Filtering by confidence

    # Indices -> N x 1 matrix
    indices = (results[0]['instances'].scores > args.confidence_threshold).nonzero()
    # Indices -> N dim vector
    indices = indices.squeeze()

    # Boxes
    results[0]['instances'].pred_boxes.tensor = torch.index_select(
        results[0]['instances'].pred_boxes.tensor, 0, indices)
    # Masks
    results[0]['instances'].pred_masks = torch.index_select(
        results[0]['instances'].pred_masks, 0, indices)
    # Classes
    results[0]['instances'].pred_classes = torch.index_select(
        results[0]['instances'].pred_classes, 0, indices)
    # Scores
    results[0]['instances'].scores = torch.index_select(
        results[0]['instances'].scores, 0, indices)

    metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

    img = img[:, :, ::-1]
    visualizer = Visualizer(img, metadata, instance_mode=ColorMode.IMAGE)
    instances = results[0]["instances"].to(torch.device("cpu"))
    # Detach box tensor to convert it into numpy arrays in visualizer
    instances.pred_boxes.tensor = instances.pred_boxes.tensor.detach()
    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    out_filename = "/home/an1/detectron2/caffe2_model/input_out.jpg"
    vis_output.save(out_filename)
