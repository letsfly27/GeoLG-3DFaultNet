import torch
import torch.nn.functional as F
import numpy as np


def dice_score(pred, label, smooth=1e-6):
    pred_prob = F.softmax(pred, dim=1)[:, 1]
    pred_binary = (pred_prob > 0.5).float()
    intersection = (pred_binary * label).sum()
    union = pred_binary.sum() + label.sum()
    return (2. * intersection + smooth) / (union + smooth)


def iou_score(pred, label, smooth=1e-6):
    pred_prob = F.softmax(pred, dim=1)[:, 1]
    pred_binary = (pred_prob > 0.5).float()
    intersection = (pred_binary * label).sum()
    union = pred_binary.sum() + label.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall(pred, label, smooth=1e-6):
    pred_prob = F.softmax(pred, dim=1)[:, 1]
    pred_binary = (pred_prob > 0.5).float()
    tp = (pred_binary * label).sum()
    fp = pred_binary.sum() - tp
    fn = label.sum() - tp
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


def accuracy_score(pred, label):
    pred_class = pred.argmax(dim=1)
    correct = (pred_class == label.long()).float().sum()
    total = torch.numel(label)
    return correct / total


def f1_score(pred, label, smooth=1e-6):

    return dice_score(pred, label, smooth)


def soft_erode(img):

    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):

    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_=5):

    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for i in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(y_pred, y_true, iter_=3, smooth=1e-5):


    skel_pred = soft_skel(y_pred, iter_)
    skel_true = soft_skel(y_true, iter_)


    tprec = (skel_pred * y_true).sum() / (skel_pred.sum() + smooth)


    tsens = (skel_true * y_pred).sum() / (skel_true.sum() + smooth)

    cl_dice = 2 * tprec * tsens / (tprec + tsens + smooth)
    return 1.0 - cl_dice

def edge_smoothing_loss(pred, target):
    pred_sigmoid = torch.sigmoid(pred)

    pred_edge = pred_sigmoid - F.avg_pool3d(pred_sigmoid, kernel_size=3, padding=1, stride=1)
    target_edge = target - F.avg_pool3d(target, kernel_size=3, padding=1, stride=1)
    return F.mse_loss(pred_edge, target_edge)


class GeoLGFaultLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, weight_cldice=0.5, weight_edge=0.5):
        super(GeoLGFaultLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.w_bce = weight_bce
        self.w_dice = weight_dice
        self.w_cldice = weight_cldice
        self.w_edge = weight_edge

    def forward(self, pred, target):

        l_bce = self.bce_loss(pred, target)
        

        l_dice = 1.0 - dice_score(pred, target)
        

        l_cldice = cldice_loss(pred, target)
        

        l_edge = edge_smoothing_loss(pred, target)
        

        total_loss = (self.w_bce * l_bce) + (self.w_dice * l_dice) + \
                     (self.w_cldice * l_cldice) + (self.w_edge * l_edge)
                     
        return total_loss