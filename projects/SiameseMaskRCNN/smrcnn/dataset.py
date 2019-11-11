import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


def get_xview2_dicts(data_dir, split):
    ann_path = os.path.join(data_dir, 'annotations', split + '.json')
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


data_dir = "/media/ashwin/DATA/detectron2/projects/SiameseMaskRCNN/data/xview2/"
for d in ["train", "val"]:
    DatasetCatalog.register("xView2_" + d, lambda d=d: get_xview2_dicts(data_dir, d))
    MetadataCatalog.get("xView2_" + d).set(thing_classes=["no-damage",
                                                          "minor-damage",
                                                          "major-damage",
                                                          "destroyed"])
