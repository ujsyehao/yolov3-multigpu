import glob
import random
import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # use center padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # add padding, pad format: (padding_left, padding_right, padding_top, padding_bottom)
    img = F.pad(img, pad, 'constant', value=pad_value) 
    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

# not used
def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest") # images: tensor
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path)) # return img list: (img1, img2 ...)
        self.img_size = img_size
    
    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # extract imag as pytorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # pad to square resolution
        img, _ = pad_to_square(img, 0) # note: can use mean_value
        # resize
        img = resize(img, self.img_size)
        return img_path, img

    def __len__(self):
        return len(self.files)

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        # label format: class + normalized (x_ctr, y_ctr, w, h)
        with open(list_path, 'r') as file:
            self.img_files = file.readlines() # img path list 

        # get label path list
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt") for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32 # 320
        self.max_size = self.img_size + 3 * 32 # 512
        self.batch_count = 0

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # extract img as pytorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # handle images with less than 3-dimension
        if len(img.shape) != 3:
            img = img.unsqueeze(0) # add 0-dimension
            img = img.expand((3, img.shape[1:])) # copy 1-channel to 3-channel

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # pad to square
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # label
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = None
        if os.path.exists(label_path):
            # class label + normalized x_ctr, y_ctr, w, h
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # get unnormalized x1, y1, x2, y2
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # adjust for added padding
            x1 += pad[0]
            x2 += pad[1]
            y1 += pad[2]
            y2 += pad[3]
            # returns class label + normalized padding image(x_ctr, y_ctr, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w # w x w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            # 6-d tensor, 0-d don't use -> used in collate_fn()
            # 0-d : batch-index
            targets[:, 1:] = boxes

        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        #print ('debug: ', img.size(), targets.size())
        return img_path, img, targets

    def collate_fn(self, batch):
        # imgs is tuple, len(imgs) = sum_batchsize
        # targets is tuple, len(targets) = sum_batchsize
        paths, imgs, targets = list(zip(*batch)) # zip(*): uncompressed
                                                 # convert list to tuple
                                                 # list(): return list

        """ 
        # remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # add batch-index to targets ex: batch size 8 -> batch-index: 0, 0, 1, 2, 3 ... 7
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i 
        targets = torch.cat(targets, 0) # 0-dimension
        """ 

        # note: solve multi gpu train problem, refer to https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/181
        # find max number of targets in one image
        max_targets = 0
        for i in range(len(targets)):
            # exist no target
            if targets[i] is None:
                continue
            length_target = targets[i].size(0)
            if (max_targets < length_target):
                max_targets = length_target

        #print ('max_targets: ', max_targets)
        new_targets = []

        for i, boxes in enumerate(targets):
            if boxes is None:
                continue
            boxes[:, 0] = i
            if (boxes.size(0) < max_targets):
                append_size = max_targets - boxes.size(0)
                append_tensor = torch.zeros((append_size, 6))
                boxes = torch.cat((boxes, append_tensor), 0)
            new_targets.append(boxes)

            #print (i, boxes)
                
        #targets = [boxes for boxes in targets if boxes is not None]
        targets = [boxes for boxes in new_targets if boxes is not None]
        targets = torch.cat(targets, 0)

        # select new image size every 10 batch 
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # resize image(pad-to-square) to new size
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets 

    def __len__(self):
        return len(self.img_files)
