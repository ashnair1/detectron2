## Siamese Mask RCNN in Detectron2

### Train
```bash
python projects/SiameseMaskRCNN/train_net.py --config-file projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```