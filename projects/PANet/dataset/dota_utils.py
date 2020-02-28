import os
import json
import math
import numpy as np
from PIL import Image
from shapely import geometry
from tqdm import tqdm

from detectron2.structures import BoxMode


def get_dota_annotation(label):

    categories = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", 
                  "ground-track-field", "harbor", "bridge", "small-vehicle", "large-vehicle", "helicopter", 
                  "roundabout", "soccer-ball-field", "swimming-pool", "container-crane"]
    catmap = {k: v for v, k in enumerate(categories)}

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
            x = vertex_coords[0, :]
            y = vertex_coords[1, :]
            xc, yc = np.mean(x), np.mean(y)
            # Find largest side in order to calculate angle between this side and x axis
            # By convention, we consider largest side as width of the object
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

            angle = math.degrees(angle) * -1
            anndict = {'bbox': [xc, yc, w, h, angle], 'bbox_mode': BoxMode.XYWHA_ABS, 'category_id': catmap[p[-2]]}
            anns.append(anndict)
        return anns


def d2_compatible_dota(root_dir):

    image_dir    = os.path.join(root_dir, 'images')
    label_dir    = os.path.join(root_dir, 'labelTxt-v1.5')
    images       = sorted([os.path.join(image_dir, b) for b in os.listdir(image_dir)])
    labels       = sorted([os.path.join(label_dir, l) for l in os.listdir(label_dir)])
    imlen        = len(images)
    lablen       = len(labels)
    
    assert imlen == lablen, "No. of images {} not equal to no. of labels {}".format(imlen, lablen)
    print("Length of dataset:{}".format(imlen))
    outjson = "dota_" + os.path.basename(root_dir) + ".json"

    outfile = os.path.join(root_dir, outjson)
    oann = {}
    for im, lbl in tqdm(zip(images, labels), total=len(images)):
        img = Image.open(im)
        w, h = img.size
        anns = get_dota_annotation(lbl)

        oann[os.path.basename(im)] = {'height': h, 'width': w, 'file_name': im, 
                                      'image_id': os.path.basename(im)[:-4], 'annotations': anns}

    with open(outfile, 'w') as f:
        json.dump(oann, f)


if __name__ == "__main__":
    data_dir = "/home/an1/detectron2/datasets/dota_800/val"
    d2_compatible_dota(data_dir)
