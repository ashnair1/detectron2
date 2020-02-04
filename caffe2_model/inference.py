from caffe2.proto import caffe2_pb2
import numpy as np
from PIL import Image
from matplotlib import pyplot
import os
from caffe2.python import core, workspace, models
import torch

IMAGE_LOCATION = "/home/an1/detectron2/caffe2_model/input.jpg"
INIT_NET = '/home/an1/detectron2/caffe2_model/model_init.pb'
PREDICT_NET = '/home/an1/detectron2/caffe2_model/model.pb'

# Read single image
img = Image.open(IMAGE_LOCATION)
img = np.array(img)

# Convert HWC -> CHW
img = img.swapaxes(1, 2).swapaxes(0, 1)

# Convert CHW -> NCHW
img = np.array([img])

# Im info N x 3 tensor of (height, width, scale)
im_info = np.reshape(np.array([np.float(img.shape[2]), np.float(img.shape[3]), 1.0]), (1,-1))
im_info = im_info.astype('float32')

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

# Initialise the predictor from the input protobufs
p = workspace.Predictor(init_net, predict_net)

# Run the net and return predictions
results = p.run({'data': img, "im_info": im_info})

import pdb
pdb.set_trace()