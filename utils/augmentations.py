import torch
import torch.nn.functional as F 
import numpy as np  

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1]) # image shape: chanels, h, w
                                      # flip on w-dimension
    # x_ctr
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
