# yolov3-multigpu
## train pytorch-yolov3 with multi GPU

This repository is forked from https://github.com/eriklindernoren/PyTorch-YOLOv3, I modify it to support multi GPU train.

If you want to use multi GPU train, you should modify these below:
* pad targets in utils/datasets.py -> collate_fn() function line149
* remove pad targets in utils/utils.py -> build_targets() function line 301
* re-calculate batch-index(targets) in utils/utils.py -> build_targets() function line 325
* move model outputs from cpu to cpu in models.py -> Darknet.forward() function line 330
* calculate loss sum from all GPUs in train.py -> line 121
* modify log info in train.py -> line 130

A few imporvements compared with original repository:
* fix gt box may cross boundary in utils/utils.py -> build_targets() function line 332
* fix target batch-index misorder in utils/datasets.py -> collate_fn() function line 162
