import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog


def get_dota_dicts(root_dir, split):
    anns = []
    with open(os.path.join(root_dir, 'dota_{}.json'.format(split))) as j:
        annj = json.load(j)

    for j in list(annj.values()):
        anns.append(j)

    return anns


def register_dota():
    data_dir = "/home/an1/detectron2/datasets/dota_800/"
    categories = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", 
                  "ground-track-field", "harbor", "bridge", "small-vehicle", "large-vehicle", "helicopter", 
                  "roundabout", "soccer-ball-field", "swimming-pool", "container-crane"]

    for d in ["train", "val"]:
        DatasetCatalog.register("dota1.5_" + d, lambda d=d: get_dota_dicts(data_dir + d, d))
        MetadataCatalog.get("dota1.5_" + d).thing_classes = categories


if __name__ == "__main__":
    data_dir = "/home/an1/detectron2/datasets/dota_800/"
    dota_dict = get_dota_dicts(data_dir + "val", "val")
