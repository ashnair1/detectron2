import cv2
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import numpy as np
import os
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import random

from detectron2.data import MetadataCatalog

if __name__ == "__main__":
    fig, ax = plt.subplots(1, figsize=(10,10))

    imgdir = "/home/an1/detectron2/datasets/coco/val2017"
    outdir = "/home/an1/detectron2/projects/PANet/demo/test_results"
    imgs = os.listdir(imgdir)
    img_name = random.choice(imgs)

    print("Running inference on {}".format(img_name))
    img = cv2.imread(os.path.join(imgdir, img_name))
    img = cv2.resize(img, (224, 224))

    # Display the image
    ax.imshow(np.flip(img, 2), )

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # CHW -> NCHW
    img = np.array([img])
    # Make im_info a 1 x 3 tensor
    im_info = np.reshape(np.array([img.shape[2], img.shape[3], 1]).astype('float32'), (1,-1))

    model_dir = model = '/home/an1/detectron2/projects/PANet/output/conversion'
    init_net_path = os.path.join(model_dir, 'model_init.pb')
    predict_net_path = os.path.join(model_dir, 'model.pb')

    device_opts = core.DeviceOption(caffe2_pb2.CPU)

    # Read the contents of the input protobufs into local variables
    init_net = caffe2_pb2.NetDef()
    with open(init_net_path, 'rb') as f:
        init_net.ParseFromString(f.read())
        init_net.device_option.CopyFrom(device_opts)

    predict_net = caffe2_pb2.NetDef()
    with open(predict_net_path, "rb") as f:
        predict_net.ParseFromString(f.read())
        predict_net.device_option.CopyFrom(device_opts)

    # Initialise the predictor from the input protobufs
    c2w = workspace.Predictor(init_net, predict_net)

    # Run the net and return predictions
    results = c2w.run({'data': img, "im_info": im_info})

    boxes, scores, cls, masks, _ = results

    boxes = boxes.tolist()
    scores = scores.tolist()
    classes = cls.tolist()

    count = 0

    for box, mask, score, cls in zip(boxes, masks, scores, classes):
        if score < 0.8:
            continue
        count += 1
        x1, y1, x2, y2 = box
        mask = mask[int(cls)]
        cls_name = MetadataCatalog.get('coco_2017_train').thing_classes[int(cls)]
        cls_color = MetadataCatalog.get('coco_2017_train').thing_colors[int(cls)]
        
        w = x2 - x1
        h = y2 - y1
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x1,y1), w, h, linewidth=2, edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        # Create mask
        mnew = cv2.resize(mask, (int(w), int(h)))
        
        ret, thresh = cv2.threshold(mnew*255, 127, 255, 0)
        border = cv2.copyMakeBorder(thresh.astype('int32'), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0 )
        contours, hierarchy = cv2.findContours(border, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
        
        # Move mask to ROI position
        for cnt in contours:
            cnt[:,0][:,0] += int(x1)
            cnt[:,0][:,1] += int(y1)
        
            poly = patches.Polygon(cnt[:,0], closed=True, edgecolor='r', facecolor=np.array(cls_color)/255, alpha=0.6)
            ax.add_patch(poly)
        
        ax.text(x1, y1, "{}:{:.2f}".format(cls_name, score * 100), 
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment="center",
                color="w",
                zorder=10,)

    fig.tight_layout()
    plt.axis('off')        
    plt.savefig(os.path.join(outdir, img_name[:-4] + ".png"))

    print("{} instances detected".format(count))