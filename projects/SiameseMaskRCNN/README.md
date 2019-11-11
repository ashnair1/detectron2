## Siamese Mask RCNN in Detectron2

### Train
1. Single GPU
```bash
python projects/SiameseMaskRCNN/train_net.py \
        --config-file projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

2. Multi GPU
```bash
python tools/train_net.py --num-gpus 8 \
	--config-file projects/SiameseMaskRCNN/configs/Base-SiameseMaskRCNN-Fast-C4.yaml
```