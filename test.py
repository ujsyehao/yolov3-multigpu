from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import argparse
import tqdm

import torch
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor

    labels = []
    sample_metrics = [] # list of tuples (TP, confs, pred)

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # extract class labels
        labels += targets[:, 1].tolist() # format: (sample_index, class, x_ctr, y_ctr, w, h)
        # get normalized (xmin, ymin, xmax, ymax)
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        # get w.r. 416x416 (xmin, ymin, xmax, ymax)
        targets[:, 2:] *= img_size

        imgs = imgs.cuda()

        with torch.no_grad():
            # outputs size: [n, 10647, 85]
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres, nms_thres)
            # nms-outputs is a list, elment is tensor
            # outputs[0]: torch.size([valid_box_number, 7]) 7 is (x1, y1, x2, y2, conf, class_score, class_pred)

        # sample_metrics is a list, elem format (true_positives, pred_scores, pred_labels)
        # length of sample_metrics: batch_size + batch_size + ....
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # concat sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class