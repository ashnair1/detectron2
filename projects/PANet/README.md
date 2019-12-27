## Path Aggregation Network in Detectron2

### Train
1. Single GPU
```bash
python projects/PANet/train_net.py \
        --config-file projects/PANet/configs/Base-PANet-FPN.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

2. Multi GPU
```bash
python projects/PANet/train_net.py --num-gpus 8 \
        --config-file projects/PANet/configs/PANet-FPN.yaml
```
3. Resume Training
```bash
python projects/PANet/train_net.py \
        --resume \
        --config-file projects/PANet/configs/Base-PANet-FPN.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

### Evaluate
```bash
python projects/PANet/train_net.py \
        --eval-only \
        --config-file projects/PANet/configs/Base-PANet-FPN.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```