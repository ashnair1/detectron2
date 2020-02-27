from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import copy
import json
import logging
import math
import numpy as np
from PIL import Image
import random
from shapely import geometry
import torch
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
    
def get_dota_annotation(label, imdim, catmap):

    if os.path.isfile(label):
        with open(label, 'r') as file:
            lines = file.read().splitlines()

    anns = []

    if len(lines) >= 1:
        invalids = 0
        for line in lines:
            p = line.split(" ")
            coords  = [(float(x), float(y)) for x, y in zip(iter(p[:-2][::2]), iter(p[:-2][1::2]))]
            quad    = geometry.Polygon(coords)
            rbbox   = quad.minimum_rotated_rectangle
            if rbbox.geom_type == "LineString":
                invalids += 1
                continue

            vertex_coords = np.array(rbbox.exterior.coords.xy)[:,:4]
            # Absolute values
            vertex_coords[0,:] = vertex_coords[0, :] / imdim[1]
            vertex_coords[1,:] = vertex_coords[1, :] / imdim[0]
            x = vertex_coords[0,:]
            y = vertex_coords[1,:]
            xc, yc = np.mean(x), np.mean(y)
            # now we need to find largest side in order to calculate angle between this side and x axis
            # by convention we consider largest side as width of the object
            d01 = np.sqrt(np.sum((vertex_coords[:, 0] - vertex_coords[:, 1]) ** 2))
            d03 = np.sqrt(np.sum((vertex_coords[:, 0] - vertex_coords[:, 3]) ** 2))
            
            if d01 > d03:
                if x[0] == x[1]:
                    angle = math.pi / 2
                else:
                    angle = math.atan((y[0] - y[1]) / (x[0] - x[1]))
                h = d03
                w = d01
            else:
                if x[0] == x[3]:
                    angle = math.pi / 2
                else:
                    angle = math.atan((y[0] - y[3]) / (x[0] - x[3]))
                h = d01
                w = d03

            angle = math.degrees(angle) 
            anndict = {'bbox': [xc, yc, w, h, angle],
                       'bbox_mode':BoxMode.XYWHA_ABS, 'category_id':catmap[p[-2]]}
            anns.append(anndict)
        return anns

    else:
        return None  

def get_dota_annotation2(label, imdim, catmap):

    if os.path.isfile(label):
        with open(label, 'r') as file:
            lines = file.read().splitlines()

    anns = []

    if len(lines) >= 1:
        invalids = 0
        for line in lines:
            a = line.split(" ")
            xc, yc, w, h, angle = [float(i) for i in a[:-1]]
            cat = a[-1]
            anndict = {'bbox': [xc, yc, w, h, angle * -1],
                       'bbox_mode':BoxMode.XYWHA_ABS, 'category_id':catmap[cat]}
            anns.append(anndict)
        return anns

    else:
        return None  
    

def get_dota_dicts(root_dir, categories):

    image_dir    = os.path.join(root_dir, 'images')
    label_dir    = os.path.join(root_dir, 'labelTxt-v1.5_d2')
    images       = sorted([os.path.join(image_dir, b) for b in os.listdir(image_dir)])
    labels       = sorted([os.path.join(label_dir, l) for l in os.listdir(label_dir)])
    imlen        = len(images)
    lablen       = len(labels)
    
    assert imlen == lablen, "No. of images {} not equal to no. of labels {}".format(imlen, lablen)
    
    print("Length of dataset:{}".format(imlen))
    
    category_map = {categories[i] : i for i in range(len(categories))}
    anns = []
    for im, lbl in tqdm(zip(images, labels), total=len(images)):
        ann = {}

        img = Image.open(im)
        w, h = img.size

        ann['height']        = h
        ann['width']         = w
        ann['image_id']      = os.path.basename(im)[:-4]
        ann['file_name']     = im
        ann['annotations']   = get_dota_annotation2(lbl, (h,w), category_map)

        if ann['annotations'] is None:
            continue

        anns.append(ann)

    return anns

def get_dota_dicts2(root_dir, cats):
    anns = []
    with open(os.path.join(root_dir, 'dota_800_val.json')) as j:
        annj = json.load(j)

    catmap = {k:v for v,k in enumerate(cats)}

    for j in list(annj.values()):
        anns.append(j)

    return anns

def register_dota():
    data_dir = "/home/an1/detectron2/datasets/dota_800/"
    categories = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", 
                  "ground-track-field", "harbor", "bridge", "small-vehicle", "large-vehicle", "helicopter", 
                  "roundabout", "soccer-ball-field", "swimming-pool", "container-crane"]

    for d in ["train", "val"]:
        DatasetCatalog.register("dota1.5_" + d, lambda d=d: get_dota_dicts(data_dir + d, categories))
        MetadataCatalog.get("dota1.5_" + d).thing_classes = categories


if __name__ == "__main__":
    data_dir = "/home/an1/detectron2/datasets/dota_800/"
    categories = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", 
                  "ground-track-field", "harbor", "bridge", "small-vehicle", "large-vehicle", "helicopter", 
                  "roundabout", "soccer-ball-field", "swimming-pool", "container-crane"]
    dota_dict = get_dota_dicts2(data_dir + "val", categories)
    import pdb
    pdb.set_trace()
