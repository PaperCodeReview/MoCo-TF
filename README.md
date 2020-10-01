# MoCo-TF

This is an unofficial implementation of Moco v1 [(Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020.)](https://arxiv.org/abs/1911.05722) and Moco v2 [(Improved Baselines with Momentum Contrastive Learning)](https://arxiv.org/abs/2003.04297).  
  
## Requirements
(TODO : requirements.txt and Dockerfile for the image of fixed environment.)
- python >= 3.6
- tensorflow >= 2.2
## Training
```
python main.py --task v1 --dataset imagenet --brightness 0.4 --contrast 0.4 --saturation 0.4 --hue 0.1 --data_path /path/of/your/data --gpus 0
```
## Reproducibility
- TODO
