# -*- coding: future_fstrings -*-
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg")
    parser.add_argument("--weights_path", type=str, default="/home/yehao/darknet/weights/yolo/yolov3.weights")
    parser.add_argument("--class_path", type=str, default="data/coco.names")
    parser.add_argument("--conf_thres", type=float, default=0.8)
    parser.add_argument("--nms_thres", type=int, default=0.4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpu", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--checkpoint_model", type=str, help="pytorch model")
    opt = parser.parse_args()

    device = torch.device("cuda")

    os.makedirs("output", exist_ok=True) # avoid existing dir cause OS error

    # set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval() # fix BN, Dropot param

    # class DataLoader, class ImageFolder
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size, 
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # extract class labels from file
    classes = load_classes(opt.class_path)

    Tensor = torch.cuda.FloatTensor

    imgs = []
    img_detections = []

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.cuda()

        with torch.no_grad(): # deactivate autograd
            detections = model(input_imgs)
            # input detections format: (x_ctr, y_ctr, w, h, objectness, cls_score)
            # return detections format: (x1, y1, x2, y2, objectness, cls_score, cls_label)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        imgs.extend(img_paths)
        img_detections.extend(detections)

    # bbox colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print('\nSaving images:')
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print ("(%d) Image: '%s' " %(img_i, path))

        # create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # draw detections
        if detections is not None:
            # rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2]) 
            # unique labels
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, objectness, cls_score, cls_label in detections:
                print ("label-%s %.2f %.2f %.2f %.2f %.2f" % (classes[int(cls_label)], cls_score, x1, y1, x2, y2))
                # plot box
                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_label))[0])]
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(bbox)
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_score)],
                    color="white",
                    verticalalignment = "top",
                    bbox={"color": color, "pad": 0},
                )
        
        # save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        #plt.show()
        plt.close()        



