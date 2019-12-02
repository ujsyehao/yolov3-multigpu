from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

def create_modules(module_defs):
    """construct layer from list(network) """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])] # 3
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                "conv_{}".format(module_i),
                nn.Conv2d(
                    in_channels = output_filters[-1],
                    out_channels = filters,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = pad,
                    bias = not bn,
                )
            )
            if bn:
                modules.add_module("batch_norm_{}".format(module_i), nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_{}".format(module_i), nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module("_debug_padding_{}".format(module_i), nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_{}".format(module_i), maxpool)
        
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsamle_{}".format(module_i), upsample)

        elif module_def["type"] == "route":
            # create in class Darknet
            # layer: -1, 36
            # layer: -4
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) # for next conv input_channels
            modules.add_module("route_".format(module_i), EmptyLayer())
        
        elif module_def["type"] == "shortcut":
            # create in class Darknet 
            filters = output_filters[1:][int(module_def["from"])] # for next conv input_channels
            modules.add_module("shortcut_{}".format(module_i), EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)] # (w, h) as a anchor format
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # define detectin layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(module_i), yolo_layer)
        
        # register module list and output filters
        module_list.append(modules)
        output_filters.append(filters) 
    
    return hyperparams, module_list

class Upsample(nn.Module):
    """ nn.Upsample is deprecated"""
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class EmptyLayer(nn.Module):
    """ Placeholder for 'route' and 'shortcu' layers """
    def __init__(self):
        super(EmptyLayer, self).__init__()    


class YOLOLayer(nn.Module):
    """Detection Layer"""
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5 # filter negative sapmles (matched IOU > 0.5 but not maxed IOU )
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1 # scale factor of has obj objectness_loss
        self.noobj_scale = 50 # scale factor of no_obj objectness_loss
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0 

    def compute_grid_offsets(self, grid_size, cuda=True):
        """
        get each grid (0, 0), (0, 1), (1, 0) ...
        get current normalize yolo layer anchor width, height: (a1_w, a2_w, a3_w), (a1_h, a2_h, a3_h)
        """
        # input: 416x416
        # grid_size: 13x13
        # grid size: 26x26
        # grid size: 52x52
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # calculate offsets for each grid -> get (0, 0), (0, 1), (1, 0) .
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # normalize w: anchow_w / stride  normalize h: anchor_h / stride
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1)) 
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1)) 

    def forward(self, x, targets=None, img_dim=None):
        FloatTensor = torch.cuda.FloatTensor 
        LongTensor = torch.cuda.LongTensor
        ByteTensor = torch.cuda.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        # convert predictions
        # note: NCHW format -> grid_y, grid_x
        # nx255x13x13 -> nx3x85x13x13 -> nx3x13x13x85
        # 85: tx_ctr, ty_ctr, tw, th, objectness, 80 class
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # get and parse outputs
        x = torch.sigmoid(prediction[..., 0]) # tx_ctr range: (0, 1) 
                                              # format: [batch_size, anchors, grid_y, grid_x]
        y = torch.sigmoid(prediction[..., 1]) # ty_ctr range: (0, 1)
        w = prediction[..., 2] # tw
        h = prediction[..., 3] # th
        pred_conf = torch.sigmoid(prediction[..., 4]) # objectness use sigmoid()
        pred_cls = torch.sigmoid(prediction[..., 5:]) # cls use sigmoid()
                                                      # format: [batch_size, anchors, grid_y, grid_x, cls]

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # add offset and scale with anchors 
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x # x_ctr range: (0, 13)  
        pred_boxes[..., 1] = y.data + self.grid_y # y_ctr range: (0, 13)
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w # width w.r.t current feature map dimension
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h # height w.r.t current feature map dimension

        # output shape: [1, x, 85]
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride, # get (x_ctr, y_ctr, w, h) w.r.t 416x416 
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            # calculate loss
            # (tx, ty, tw, th): target offset
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes, # normalize x_ctr, y_ctr, w, h
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors, # normalize (anchor w, anchor h) w.r. current yolo layer dimension
                ignore_thres=self.ignore_thres, # 0.5
            )

            """
            test code
            """
            tmp = list(obj_mask.size())
            sum = 1
            for item in tmp:
                sum *= item
            #print ('sum anchors: ', sum)
            #print ('positive samples: ', list(obj_mask[obj_mask].size())[0])
            #print ('negative sample: %d \n' %(list(noobj_mask[noobj_mask].size())[0]))


            # calculate loss
            #print ('loss')

            """
            calculate postive samples loss: loc loss + cls loss + obj loss
            """
            # calculate loc loss
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask]) # choose positive predict box tx ang target tx*
                                                              # x size: [batch_size, anchors, grid_y, grid_x]
                                                              # obj_mask size: [batch_size, anchors, grid_y, grid_x]
                                                              # tx size:  [batch_size, anchors, grid_y, grid_x]
                                                              # x[obj_mask] size: [14] 14 is number of positive samples
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            # calculate cls loss
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask]) # pred_cls size: [1, 3, 13, 13, 80]
                                                                         # obj_mask size: [1, 3, 13, 13]
                                                                         # pred_cls[obj_mask] size: [n, 80] 
                                                                         # tcls[obj_mask] size: [n, 80]
                                                                         # loss_cls: 1/N * Sum(-(y x logp + (1-y) x log(1-p)))

            # calculate obj loss
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask]) # tconf = obj_mask
                                                                                # tconf[obj_mask]: [1, 1, 1, 1, 1 ...] note: just choose 1(target)
                                                                                # pred_conf[obj_mask]: [0.1, 0.12, 0.13 ...] 
                                                                                # use binary cross-entropy loss

            """
            calculate negative samples loss: no obj loss
            """
            # calculate no-obj loss
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask]) # tconf = obj_mask
                                                                                      # obj_mask[noobj_mask]: just choose 0(target)
            
            """
            loss post-process
            """
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj # note: it is unreasonable
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # metrics
            cls_acc = 100 * class_mask[obj_mask].mean() # class_mask[obj_mask] size: [20] 20 is positive samples number
            conf_obj = pred_conf[obj_mask].mean() # pred_conf[obj_mask] size: [20] 20 is positve samples number
            conf_noobj = pred_conf[noobj_mask].mean() # pred_conf[noobj_mask] size: [2000] 2000 is negative samples number
            conf50 = (pred_conf > 0.5).float() # size: [1, 3, 13, 13]
            iou50 = (iou_scores > 0.5).float() # size: [1, 3, 13, 13]
            iou75 = (iou_scores > 0.5).float() # size: [1, 3, 13, 13]
            detected_mask = conf50 * class_mask * tconf # size: [1, 3, 13, 13]
                                                        # objectness > 0.5 and predict class is correct 
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16) # precision = TP / (TP + FP)
                                                                                  # TP: objectness > 0.5 && predict class correct && IOU > 0.5
                                                                                  # TP + FP: objectness > 0.5
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16) # recall = TP / (TP + FN)
                                                                                   # TP: objectness > 0.5 && predict class correct && IOU > 0.5
                                                                                   # TP + FN : all positive samples(obj_mask)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            #print (grid_size, 'x', grid_size, '-loss: ', to_cpu(total_loss).item(), ' coord loss: ', 
            #        to_cpu(loss_x).item() + to_cpu(loss_y).item() + to_cpu(loss_w).item() + to_cpu(loss_h).item(), 
            #        ' conf loss: ', to_cpu(loss_conf).item(), ' cls loss: ', to_cpu(loss_cls).item())

            self.metrics = {
                "grid_size": grid_size,
                "loss": to_cpu(total_loss).item(),
                "loss-tx": to_cpu(loss_x).item(),
                "loss-ty": to_cpu(loss_y).item(),
                "loss-tw": to_cpu(loss_w).item(),
                "loss-th": to_cpu(loss_h).item(),
                "loss-conf": to_cpu(loss_conf).item(),
                "loss-cls": to_cpu(loss_cls).item(),
                "loss-obj": to_cpu(loss_conf_obj).item(),
                "loss-noobj x scale": to_cpu(loss_conf_noobj * self.noobj_scale).item(),
                "loss-noobj": to_cpu(loss_conf_noobj).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),                
            }

            #print (self.metrics)
            self.noobj_scale = 100000

            return output, total_loss


class Darknet(nn.Module):
    """ Yolo v3 detection model """
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.test_yolo_layers = []
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        #print ('forward x: ', x.size(), type(x), x.type(), x.is_cuda, x.device)
        #print ('forward targets: ', targets.size())

        # clear list
        self.test_yolo_layers.clear()

        img_dim = x.shape[2]
        loss = 0 # sum of 3 yolo layer loss
        layer_outputs, yolo_outputs = [], [] # layer_outputs: all layers
                                             # yolo_outputs: 3 yolo layer output
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1) # 1-dimension: channels
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"]) # ex: from -3
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                self.test_yolo_layers.append(module[0])
                x, layer_loss = module[0](x, targets, img_dim) # module[0]: yolo layer
                loss += layer_loss
                yolo_outputs.append(x)

            layer_outputs.append(x) # form yolov3 layer
        

        #print (self.yolo_layers[0].metrics) # not work
        #print (self.test_yolo_layers[0].metrics)


        # [1, 507, 85], [1, 2028, 85], [1, 8112, 85] -> [1, 10647, 85]
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1)) # 1-dimension: number of pred boxes
                                                          # move tensor to cpu -> for post-process

        # note: need modify when use multi-gpu train
        if loss == 0:
            loss_ = loss
        else:
            loss_ = loss.type(torch.cuda.FloatTensor) # multi gpu train need cuda tensor

        #print ('yolo outputs', type(yolo_outputs), yolo_outputs.type(), yolo_outputs.is_cuda, yolo_outputs.device)
        #yolo_outputs_ = type(torch.cuda.FloatTensor)
        yolo_outputs_gpu = yolo_outputs.cuda()
        #print ('yolo outputs', type(yolo_outputs_), yolo_outputs_.type(), yolo_outputs_.is_cuda, yolo_outputs_.device)
        return yolo_outputs_gpu if targets is None else (loss_, yolo_outputs_gpu)

    def load_darknet_weights(self, weights_path):
        # open the weight file
        with open(weights_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5) # header values: version number ...
            self.head_info = header # needed to write header when saving weights
            self.seen = header[3] # number of trained images in Train phase
            weights = np.fromfile(f, dtype=np.float32) # rest are weights
        
        # establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0 # number of weights
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0] # conv module block: (conv, BN, leakyReLU)
                if module_def["batch_normalize"]:
                    # load BN bias, weights, running mean and running variances
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # numel(): return number of elements
                                                  # bias(): similar to beta()
                    # beta
                    beta = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(beta)
                    ptr += num_b

                    # gamma
                    gamma = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(gamma)
                    ptr += num_b

                    # running mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b

                    # running variance
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # load conv_layer.bias 
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # load conv_layer.weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            path: path of the stored weight file
            cutoff: save layers between 0 and cutoff(cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp) # write header to darknet

        # iterative through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0] # conv module block: (conv, BN, leakyReLU)
                # If batch_norm, load it first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()



        
            
