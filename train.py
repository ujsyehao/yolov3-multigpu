# -*- coding: future_fstrings -*-

from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time 
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulations", type=int, default=2)
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg")
    parser.add_argument("--data_config", type=str, default="config/coco.data")
    parser.add_argument("--pretrained_weights", type=str, default="/home/yehao/darknet53.conv.74")
    parser.add_argument("--n_cpu", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--checkpoint_interval_epoch", type=int, default=1)
    parser.add_argument("--evaluation_interval_epoch", type=int, default=1)
    parser.add_argument("--compute_map", default=False)
    parser.add_argument("--multiscale_training", default=True)
    opt = parser.parse_args()

    logger = Logger("logs")

    device = torch.device("cuda")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # load and initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal) # initialize weights


    # if specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = DataLoader(
        dataset, 
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True, # pinned memory
        collate_fn=dataset.collate_fn,
    )

    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])

    # use adam optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=float(model.hyperparams['learning_rate']), weight_decay=float(model.hyperparams['decay']))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print (optimizer) 


    metrics = [
        "grid_size",
        "loss",
        "loss-tx",
        "loss-ty",
        "loss-tw",
        "loss-th",
        "loss-conf",
        "loss-cls",
        "loss-obj",
        "loss-noobj x scale",
        "loss-noobj",        
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train() # every epoch 
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i # len(dataloader) = 1 epoch

            imgs = imgs.cuda()
            targets = targets.cuda()

            print ('imgs size: ', imgs.size())
            print ('targets size: ', targets.size())

            loss, outputs = model(imgs, targets)
            # loss.backward(torch.Tensor([1]))
            # loss.backward()
            loss.sum().backward()

            if batches_done % opt.gradient_accumulations:
                # accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            log_str = "\n---- [epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.module.yolo_layers))]]]

            for i, metric in enumerate(metrics):
                formats = {m: "%.2f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.module.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.module.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.sum().item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)
            logger.list_of_scalars_summary(tensorboard_log, batches_done)
            #log_str += AsciiTable(metric_table).table 
            log_str += f"\nTotal loss {loss.sum().item()}"

            print (log_str)
            model.module.seen += imgs.size(0) # batch_size

        if epoch % opt.evaluation_interval_epoch == 0:
            print ('\n------ Evaluating model-------')
            precision, recall, AP, f1, ap_class = evaluate(
                model, 
                path = valid_path,
                iou_thres = 0.5,
                conf_thres = 0.01,
                nms_thres = 0.5,
                img_size = opt.img_size, 
                batch_size = 8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print (AsciiTable(ap_table).table)
            print (f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval_epoch == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_%d.pth" % epoch)
