# MoCo-TF

This is an unofficial implementation of Moco v1 [(Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020.)](https://arxiv.org/abs/1911.05722) and Moco v2 [(Improved Baselines with Momentum Contrastive Learning)](https://arxiv.org/abs/2003.04297).  
  
## Requirements
- python >= 3.6
- tensorflow >= 2.2 (2.2 and 2.3)

## Training
For training moco v1,
```
python main.py \
    --task v1 \
    --weight_decay 0.0001 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.4 \
    --lr_mode exponential \
    --lr_interval 120,160 \
    --data_path /path/of/your/data \
    --gpus gpu id(s) which will be used
```
or moco v2,
```
python main.py \
    --task v2 \
    --weight_decay 0.0001 \
    --mlp \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --lr_mode cosine \
    --data_path /path/of/your/data \
    --gpus gpu id(s) which will be used
```

## Evaluation
For training linear classification,
```
python main.py \
    --task lincls \
    --batch_size 256 \
    --epochs 100 \
    --lr 30 \
    --lr_mode constant \
    --data_path /path/of/your/data \
    --snapshot /path/of/checkpoint \
    --gpus gpu id(s) which will be used
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

## Citation
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}

@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```
