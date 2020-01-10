from .pafpn import build_resnet_pafpn_backbone, PAFPN
from .adaptive_pooler import AdaptiveROIPooler
from .config import add_panet_config
from .roi_heads import ROI_BOX_HEAD_REGISTRY, ROI_MASK_HEAD_REGISTRY, ROI_HEADS_REGISTRY

# Register a dummy dataset in COCO format
from detectron2.data.datasets import register_coco_instances
register_coco_instances("dummy_train", {}, "/home/ashwin/Desktop/Projects/detectron2/datasets/dummy/annotations/train.json", "/home/ashwin/Desktop/Projects/detectron2/datasets/dummy/images/train/")