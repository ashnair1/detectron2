from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import numpy as np
from PIL import Image
import numpy as np
import random
import logging
import copy
import torch
from shapely import geometry
import math

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def quad_to_xywha(coords, imdim):
    carr  = np.array(coords)
    x, y  = carr[:,::2], carr[:,1::2]
    # Convert to absolute values
    xcoord, ycoord  = x / imdim[1], y / imdim[0]

    boxes = []
    
    for c in range(len(coords)):
        poly = geometry.Polygon([(a, b) for a, b in zip(xcoord[c], ycoord[c])])
        rbbox = poly.minimum_rotated_rectangle
        vertex_coords = np.array(rbbox.exterior.coords.xy)[:,:4]
        x = vertex_coords[0,:]
        y = vertex_coords[1,:]
        xc, yc = list(rbbox.centroid.coords)[0]
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
        boxes.append(np.array([xc, yc, w, h, angle]))

    return boxes
    

def get_dota_annotation(label, imdim, catmap):
    file = open(label, 'r')
    anns = file.readlines()

    if len(anns) >= 2:
        coordlist = []
        cats = []
        difficulty = []
        for line in anns:
            line = line.strip()
            p = line.split(" ")
            coords = [float(i) for i in p[:-2]]
            coordlist.append(coords)
            cats.append(p[-2])
            difficulty.append(int(p[-1]))
        
        xywha = quad_to_xywha(coordlist, imdim)
        
        ann = []
        for box, cat in zip(xywha, cats):
            try:
                ann.append({'bbox': box, 'bbox_mode':4, 'category_id': catmap[cat]})
            except KeyError:
                print(cat)
        return ann
    else:
        return None
    

def get_dota_dicts(root_dir, categories):

    image_dir    = os.path.join(root_dir, 'images')
    label_dir    = os.path.join(root_dir, 'labelTxt-v1.5')
    images       = sorted([os.path.join(image_dir, b) for b in os.listdir(image_dir)])
    labels       = sorted([os.path.join(label_dir, l) for l in os.listdir(label_dir)])
    imlen        = len(images)
    lablen       = len(labels)
    
    assert imlen == lablen, "No. of images {} not equal to no. of labels {}".format(imlen, lablen)
    
    print("Length of dataset:{}".format(imlen))
    
    category_map = {categories[i] : i for i in range(len(categories))}
    anns = []
    for im, lbl in zip(images, labels):
        ann = {}

        img = Image.open(im)
        w, h = img.size

        ann['height']        = h
        ann['width']         = w
        ann['image_id']      = os.path.basename(im)[:-4]
        ann['file_name']     = im
        ann['annotation']    = get_dota_annotation(lbl, (h,w), category_map)

        if ann['annotation'] is None:
            continue

        anns.append(ann)

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
    dota_dict = get_dota_dicts(data_dir + "val", categories)
    import pdb
    pdb.set_trace()