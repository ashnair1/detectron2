## Path Aggregation Network in Detectron2

Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, Jiaya Jia.

[[`arXiv`](https://arxiv.org/pdf/1803.01534)] [[`BibTeX`](#CitingPANet)]

### Train
1. Single GPU
```bash
python projects/PANet/train_net.py \
        --config-file projects/PANet/configs/panet_R_50_FPN_1x.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

2. Multi GPU
```bash
python projects/PANet/train_net.py --num-gpus 8 \
        --config-file projects/PANet/configs/panet_R_50_FPN_1x.yaml
```
3. Resume Training
```bash
python projects/PANet/train_net.py \
        --resume \
        --config-file projects/PANet/configs/panet_R_50_FPN_1x.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

### Evaluate
```bash
python projects/PANet/train_net.py \
        --eval-only \
        --config-file projects/PANet/configs/panet_R_50_FPN_1x.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```


## <a name="CitingPANet"></a>Citing PANet

If you use PANet, please use the following BibTeX entry.

```
@inproceedings{liu2018path,
      author = {Shu Liu and
                Lu Qi and
                Haifang Qin and
                Jianping Shi and
                Jiaya Jia},
      title = {Path Aggregation Network for Instance Segmentation},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2018}
    }
```