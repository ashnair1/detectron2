## Siamese Mask RCNN in Detectron2 for xView2 Building Damage Classification challenge

### Train
1. Single GPU
```bash
python projects/SiameseMaskRCNN/train_net.py \
        --config-file projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

```bash
python projects/SiameseMaskRCNN/train_net.py \
        --config-file projects/SiameseMaskRCNN/configs/SiameseMaskRCNN-FPN.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

2. Multi GPU
```bash
python projects/SiameseMaskRCNN/train_net.py --num-gpus 8 \
        --config-file projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml
```
3. Resume Training
```bash
python projects/SiameseMaskRCNN/train_net.py \
        --resume \
        --config-file projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

### Evaluate
```bash
python projects/SiameseMaskRCNN/train_net.py \
        --eval-only \
        --config-file projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```