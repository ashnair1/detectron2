import os
import json
import numpy as np

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

    # Set seed for palette colour
    np.random.seed(121)
    cat_colours = []
    for c in range(len(categories)):
        colour = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
        cat_colours.append(colour) 


    for d in ["train", "val"]:
        DatasetCatalog.register("dota1.5_" + d, lambda d=d: get_dota_dicts(data_dir + d, d))
        MetadataCatalog.get("dota1.5_" + d).thing_classes = categories
        MetadataCatalog.get("dota1.5_" + d).thing_colors = cat_colours
        MetadataCatalog.get("dota1.5_" + d).json_file = data_dir + d + os.sep + "dota_{}.json".format(d)


if __name__ == "__main__":
    data_dir = "/home/an1/detectron2/datasets/dota_800/"
    dota_dict = get_dota_dicts(data_dir + "val", "val")
