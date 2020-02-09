from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_isaid():
    # Register iSAID dataset (in COCO format)
    register_coco_instances("isaid_train", {},
                            "/home/an1/detectron2/datasets/isaid/annotations/instances_train.json", 
                            "/home/an1/detectron2/datasets/isaid/images/train/")
    register_coco_instances("isaid_val", {},
                            "/home/an1/detectron2/datasets/isaid/annotations/instances_val.json", 
                            "/home/an1/detectron2/datasets/isaid/images/val/")

def register():
    register_isaid()
