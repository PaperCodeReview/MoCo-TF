# MoCo-TF

This is an unofficial implementation of Moco v1 [(Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020.)](https://arxiv.org/abs/1911.05722) and Moco v2 [(Improved Baselines with Momentum Contrastive Learning)](https://arxiv.org/abs/2003.04297).  
  
## Requirements
(**TODO** : requirements.txt and Dockerfile for the image of fixed environment.)
- python >= 3.6
- tensorflow >= 2.2

## Training
For training moco v1,
```
python main.py \
    --task v1 \
    --dataset imagenet \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.4 \
    --data_path /path/of/your/data \
    --gpus 0
```
or moco v2,
```
python main.py \
    --task v2 \
    --dataset imagenet \
    --mlp 128 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --lr_mode cosine \
    --data_path /path/of/your/data \
    --gpus 0
```

## Evaluation
**TODO**
```
```

## Results
Our model achieves the following performance on :
### Image Classification on ImageNet (IN-1M)
#### MoCo v1
|         Model         | batch | Accuracy (paper) | Accuracy (ours) |
| --------------------- | ----- | ---------------- | --------------- |
| ResNet50 (200 epochs) |  256  |       60.6       |       -         |
  
#### MoCo v2
|         Model         | batch | Accuracy (paper) | Accuracy (ours) |
| --------------------- | ----- | ---------------- | --------------- |
| ResNet50 (200 epochs) |  256  |       67.5       |        -        |
| ResNet50 (800 epochs) |  256  |       71.1       |        -        |

