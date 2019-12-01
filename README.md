# yolov3-multigpu
## train pytorch-yolov3 with multi GPU

This repository is forked from https://github.com/eriklindernoren/PyTorch-YOLOv3, I modify it to support multi GPU train.

If you want to use multi GPU train, you should modify these below:
* pad targets in utils/datasets.py -> collate_fn() function line149
* remove pad targets in utils/utils.py -> build_targets() function line 301
* re-calculate batch-index(targets) in utils/utils.py -> build_targets() function line 325
* move model outputs/loss from cpu to gpu in models.py -> Darknet.forward() function line 330
* calculate loss sum from all GPUs in train.py -> line 121
* modify log info in train.py -> line 130




## 原来仓库问题修复:
1. mAP计算
论文中是55.3, 原来仓库使用conf_thresh=0.001@IOU=0.5 mAP为53.6
* 修改datasets.py中collate_fn(), 具体原因可查看https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/243 -> mAP为54.9
* 修改utils.py中get_batch_statistics(), 具体原因可查看https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/233 -> mAP为52.6
* 原来仓库作者提供的coco2014 val 5k dataset和coco官网提供的5k val dataset不一致

2. gt box越界错误
fix gt box may cross boundary in utils/utils.py -> build_targets() function line 332

3. 多卡读取weights

４．多卡log信息

## TODO tricks:
### training tricks
* data augmentation
* multi scale train √
* multi GPU train √
* label smooth 
* mix up 
* cosine lr √
* warmup √
* group normalization (deprecated)
### detection tricks
* focal loss 
* soft nms (not supported)
* GIOU 
