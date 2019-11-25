from __future__ import division
import math 
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 

def to_cpu(tensor):
    # conver to node which doesn't need gradient
    return tensor.detach().cpu()

def load_classes(path):
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1] # not include last name
    return names

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # normal distribution: (mean, std)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # normal distribution
        torch.nn.init.constant_(m.bias.data, 0.0)

def rescale_boxes(boxes, current_dim, original_shape):
    # current_dim: 416x416 
    # original_shape: dst img shape
    # dst img -> center padding -> resize to 416x416
    orig_h, orig_w = original_shape
    # get 416x416 pad_width, pad_height
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # get 416x416 unpad_width, unpad_height
    unpad_w = current_dim - pad_x
    unpad_h = current_dim - pad_y
    # rescale bbox to original shape: normalize_coord * orig_coor
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes 

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y 

def get_batch_statistics(outputs, targets, iou_threshold):
    """
    Compute TP, predicted scores and predicted class label per sample (TP definition: predict class is correct && IOU > iou_threshold && conf > conf_thresh(in NMS))
    outputs format: list[tensor] tensor.size(valid_boxes, 7) 7 is (x1, y1, x2, y2, object_conf, class_score, class_pred)
    target format: [gt_boxes_num, 6] 6 is (sample_index, class, x1, y1, x2, y2)
    all (x1, y1, x2, y2) w.r.t 416x416
    """
    batch_metrics = []
    # calculate on each image
    for sample_i in range(len(outputs)):
        # sample_i: batch-index
        if outputs[sample_i] is None:
            continue
        
        output = outputs[sample_i] # tensor size: (valid_pred_boxes, 7)
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        # true_positives shape: (valid_boxes, ) valid_boxes w.r.t current batch-index predict boxes
        true_positives = np.zeros(pred_boxes.shape[0])

        # annotations shape: [gt_boxes, 5] gt_boxes w.r.t current batch-index / 5 is (class, x1, y1, x2, y2)
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        # get target class labels: [gt_boxes]
        target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):
            # put matched gt box index
            detected_boxes = []
            # get target boxes: [gt_boxes, 4] / 4 is (x1, y1, x2, y2)
            target_boxes = annotations[:, 1:] 

            # calculate on each predict_valid box
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                # if targets are all found(recall=1) -> break
                if len(detected_boxes) == len(annotations):
                    break

                # ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                
                # calculate pred_box and all gt boxes IOU
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0) # iou size: [gt_boxes_number]
                                                                                      # box_index: best matched gt box index
                # TP definition: predict class is correct && IOU > iou_threshold && conf > conf_thresh(in NMS)                                                                      
                if iou >= iou_threshold and box_index not in detected_boxes and pred_label == target_labels[box_index]:
                #if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def compute_ap(recall, precision):
    """
    # Arguments:
        recall: recall curve(list)
        precision: precision curve(list)
    # Returns
        average precision as computed in py-faster-rcnn
    """
    # AP use new_method(calculate all points)

    # first append sentinel values at the front and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute precision -> max Precision(R > R') R' is list
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # calculate area under RP curve, look for points where recall changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum area = w x h = difference between adjacent recall values x max Precision(R > R')
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i+1])

    return ap    

def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    compute AP on each class
    """
    # sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # find unique classes
    unique_classes = np.unique(target_cls)

    # create P-R curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc='Computing AP'):
        i = pred_cls == c # get predict class correct boxes
        n_gt = (target_cls == c).sum() # number of class-c gt boxes
        n_p = i.sum() # number of class-c predict boxes

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # accumulate FPs and TPs
            # note: [1, 2, 3, 4, 5] -> [1, 3, 6, 10, 15]
            fpc = (1 - tp[i]).cumsum() # format: list 
            tpc = (tp[i]).cumsum() # format: list

            # calculate recall
            recall_curve = tpc / (n_gt + 1e-16)
            # calculate precision
            precision_curve = tpc / (tpc + fpc)

            # AP for P-R curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # precision, recall for F1-score
            r.append(recall_curve[-1]) # just add top-N w.r.t recall
            p.append(precision_curve[-1]) # just add top-N w.r.t precision
    
    # compute F1 score 
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    
    return p, r, ap, f1, unique_classes.astype('int32')


"""
It is computed on one cell in the feature map, the centers of anchors and gt boxes are the same
note: can refer to https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/270
"""
def bbox_wh_iou(wh1, wh2):
    # wh1: (width, height)
    # wh2: (width, height)
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area 

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # transform (x_ctr, y_ctr, w, h) to (x1, y1, x2, y2)
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(
        inter_y2 - inter_y1 + 1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
    return iou 


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    prediction format: (n, d), d: (x_ctr, y_ctr, w, h, objectness, class)
    Removes detections with lower objectness than conf thresh
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # conver (x_ctr, y_ctr, w, h) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    # just execute once
    for image_i, image_pred in enumerate(prediction):
        # image_i, image_pred ex: 0, tensor([10647, 85]) 7, tensor([10647, 85])

        # filter out objectness scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # if all predictions are filtered -> process next image
        if not image_pred.size(0):
            continue
        max_class_confidence = image_pred[:, 5:].max(1)[0] # max(1): get max class (values, indices)
                                                           # [0]: get max class values
        # category confidence = objectness x max_class_confidence                                                         
        score = image_pred[:, 4] * max_class_confidence
        # sort score descend
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True) # get max class confidence and class label
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # perform NMS
        keep_boxes = []
        while detections.size(0):
            # unsqueeze(0): add dimension 
            # return (0, 0, 1, 0 ...)
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # return (0, 1, 0, 0 ...)
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower category confidence, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5] # objectness score
            # note: new_detection_box coord = merge overlapping(eliminated) box coord 
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid] # go to next iteration
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor # 8-bit unsigned tensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    # ex: box size: [batch, 3, 13, 13, 4]
    nB = pred_boxes.size(0) # batch size
    nA = pred_boxes.size(1) # number of anchors
    nG = pred_boxes.size(2) # grid_size
    # ex: cls Size: [batch, 3, 13, 13, 80]
    nC = pred_cls.size(-1) # cls number

    #print ('##########  %dx%d ##########' %(nG, nG))

    # initialize output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # initialize to 0 -> choose has_obj
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # initialize to 1 -> choose no_obj
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0) # note: whether the predict class is correct
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0) # note: target objectness score = Pr(object) * IOU = 1 * IOU = IOU
    tx = FloatTensor(nB, nA, nG, nG).fill_(0) # target x_ctr
    ty = FloatTensor(nB, nA, nG, nG).fill_(0) # target y_ctr
    tw = FloatTensor(nB, nA, nG, nG).fill_(0) # target width
    th = FloatTensor(nB, nA, nG, nG).fill_(0) # target height
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0) # target class

    obj_mask = obj_mask.bool()
    noobj_mask = noobj_mask.bool()

    # note: solver multi gpu problem
    #print ('target size: ', target.size())
    #print (target.sum(dim=1))
    target  = target[target.sum(dim=1) != 0]
    #print ('after remove fill targets size: ', target.size())
    #print (target.sum(dim=1))



    # target shape: [index, class, x_ctr, y_ctr, w, h] (normalized)
    target_boxes = target[:, 2:6] * nG # range (0, 1) -> range(0, 13)
                                       # get target coordinates w.r. current feature map dimension
    gxy = target_boxes[:, :2] 
    gwh = target_boxes[:, 2:]

    # note: calculate IOU between anchor box and gt box
    # anchor box ang gt box has the same cell, the center are the same
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]) # return tensor shape: (3, n) n is gt box number
    #print (ious)

    best_ious, best_n = ious.max(0) # get each gt box matched best anchor
                                    # return shape: (1, n)


    b, target_labels = target[:, :2].long().t() # b: batch-size index

    # note: need modify batch-index: batch-index = batch-index % imgs_per_gpu
    b = b % 8

    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t() # get (gx, gy) int indice (convert float to integer)

    # note: gt box may cross the boundary
    gi[gi < 0] = 0
    gj[gj < 0] = 0
    gi[gi > nG - 1] = nG - 1
    gj[gj > nG - 1] = nG - 1


    #print ('####: ', b, best_n, gj, gi)
    #print ('obj_mask: ', obj_mask.size())

    # according to gt indice, set mask
    obj_mask[b, best_n, gj, gi] = 1 # why use gj, gi rather than gi, gj? because pytorch tensor format [N, C, H, W]
                                    # so (y, x) -> must be the same with prediction format [N, H, W, 3, 85]
                                    # note: can decrease positive samples, when 2 gt box center are in the same grid cell and
                                    # match the same index anchor, it will decrease positive samples
    #print ('modify obj_mask: ', obj_mask.size())
    noobj_mask[b, best_n, gj, gi] = 0 # has-obj: 0, no-obj: 1

    a1 = list(obj_mask[obj_mask].size())[0]
    a2 = list(target_boxes.size())[0]
    b1 = list(noobj_mask[noobj_mask].size())[0]

    # set noobj mask to 0(has-obj) where iou > ignore_threshold
    for i, anchor_ious in enumerate(ious.t()):
        # ious.t() shape: [number of gt boxes, 3]
        # i: 0, 1, 2 ...
        # anchor_ious: tensor(iou1, iou2, iou3)
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0 # i is index
                                                                       # b[i]: get batch_size index of gt box i
                                                                       # note: can decrease negative samples, when the anchor matched gt box 
                                                                       # IOU > 0.5, it is neither positive samples nor negative samples

    b2 = list(noobj_mask[noobj_mask].size())[0]

    """
    test code
    """
    """
    if (a1 != a2 and b1 != b2):
        #for item in zip(best_n, gj, gi):
        #    print (item)
        print ('positive anchors filter: %d -> %d' %(a2, a1))
        print ('negative anchors filter: %d -> %d' %(b1, b2))    
    else :
        print ('positive anchors filter: %d -> %d' %(a2, a1))
        print ('negative anchors filter: %d -> %d' %(b1, b2))  
    
    print ('\n')
    """
                                                

    # calculate target offsets
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # one-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # compute label correctness
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # calculate iou scores
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    # tconf
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
    












    




            





