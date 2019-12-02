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

def adjust_learning_rate(optimizer, epoch):
    # use warmup
    if epoch < 5:
        lr = opt.lr * ((epoch + 1) / 5)
    else:
    # use cosine lr
        PI = 3.14159
        lr = opt.lr * 0.5 * (1 + math.cos(epoch * PI / opt.epochs)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulations", type=int, default=2)
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg")
    parser.add_argument("--data_config", type=str, default="config/coco.data")
    parser.add_argument("--pretrained_weights", type=str, default="/home/yehao/darknet53.conv.74")
    parser.add_argument("--n_cpu", type=int, default=64)
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

    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])

    # if specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.module.load_state_dict(torch.load(opt.pretrained_weights)) #note: multi-gpu train should use model.moudle
        else:
            model.module.load_darknet_weights(opt.pretrained_weights) # note: multi-gpu train should use model.modulee
            #model.load_darknet_weights(opt.pretrained_weights) # note: multi-gpu train should use model.modulee

    # warning: if first loads weights then use DataParallel() -> it will not load weights
    #model = nn.DataParallel(model, device_ids=[0,1]) 

    # get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = DataLoader(
        dataset, 
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True, # pinned memory
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    # use adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    #optimizer =torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-5) # note: which can cause error
                                                                                                # device aassert failure
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

    formats = {m: "%.2f" for m in metrics}
    #formats["grid_size"] = "%2f"
    #formats["cls_acc"] = "%.2f"

    for epoch in range(opt.epochs):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch)

        model.train() # every epoch 
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i # len(dataloader) = 1 epoch

            imgs = imgs.cuda()
            targets = targets.cuda()

            print ('imgs size: ', imgs.size())
            print ('targets size: ', targets.size())
            for param_group in optimizer.param_groups:
                print('learning rate: ', param_group['lr'])
            
            loss, outputs = model(imgs, targets)
            # loss.backward(torch.Tensor([1]))
            # loss.backward()
            #loss.sum().backward() # note: error
            loss.mean().backward()

            if batches_done % opt.gradient_accumulations:
                # accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            log_str = "---- [epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.module.test_yolo_layers))]]]
            metric_table1 = [["Metrics", *[f"YOLO Layer {i}" for i in range(3)]]]

            #print ('sum yolo layers: ', len(model.module.test_yolo_layers)) # 3 yolo layers

            # log test
            min_grid_size = int(imgs.size(2) / 32)
            medium_grid_size = int(imgs.size(2) / 16)
            max_grid_size = int(imgs.size(2) / 8)
            #print ('grid_size: ', min_grid_size, medium_grid_size, max_grid_size)

            min_status = False
            medium_status = False
            max_status = False

            for i, metric in enumerate(metrics):
                formats = {m: "%.2f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f"
                #print (model.module.test_yolo_layers[2].metrics)
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.module.test_yolo_layers]    
                metric_table += [[metric, *row_metrics]]

            # modify metric_table
            #print (type(metric_table), len(metric_table))
            #print (metric_table[2][0], metric_table[2][-1])

            gpu_number = len(model.module.test_yolo_layers) / 3.
            # sum min_grid_size loss in log

            for j, metric in enumerate(metrics):
                j = j + 1
                min = 0
                medium = 0
                max = 0
                for i in range(len(model.module.test_yolo_layers)):
                    i = i + 1
                    #print ('debug: ', metric_table[1][i], min_grid_size)
                    if int(metric_table[1][i]) == min_grid_size:
                        min += float(metric_table[j][i])
                        #print ('loss_min: ', metric_table[2][i])
                    if int(metric_table[1][i]) == medium_grid_size:
                        medium += float(metric_table[j][i])
                    if int(metric_table[1][i]) == max_grid_size:
                        max += float(metric_table[j][i])

                min = round(min / gpu_number, 2)
                medium = round(medium / gpu_number, 2)
                max = round(max / gpu_number, 2)                   
                metric_table1 += [[metric, min, medium, max]]

            # update tensorboard log
            tensorboard_log = []
            for i in range(3):    
                j = 0
                for name, metric in model.module.test_yolo_layers[0].metrics.items():
                    j += 1
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{i+1}", metric_table1[j][i+1])]

            tensorboard_log += [("loss", loss.mean().item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)


            #logger.list_of_scalars_summary(tensorboard_log, batches_done)
            log_str += AsciiTable(metric_table1).table 
            log_str += f"\nTotal loss {loss.mean().item()}\n"
            print (log_str)
            model.module.seen += imgs.size(0) # batch_size

        if epoch % opt.checkpoint_interval_epoch == 0:
            torch.save(model.module.state_dict(), f"checkpoints/yolov3_epoch%d.pth" % epoch) # save single-gpu format
        
        if epoch % opt.evaluation_interval_epoch == 0:
            print ('\n------ Evaluating model-------')
            precision, recall, AP, f1, ap_class = evaluate(
                model, 
                path = valid_path,
                iou_thres = 0.5,
                conf_thres = 0.01,
                nms_thres = 0.5,
                img_size = opt.img_size, 
                batch_size = 8*2,
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

            # write mAP to log
            file = open('./mAP.txt', 'a+')
            file.write('epoch_' + str(epoch + 1) + ' ' + str(AP.mean()) + '\n')

