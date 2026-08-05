from copy import copy, deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from scipy.ndimage import label
from PIL import Image, ImageDraw
import imageio.v2 as imageio

import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F


def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class Criterion(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(
            criterion, BaseCriterion
        ), f"{criterion} is not a proper criterion!"
        self.criterion = copy(criterion)

    def get_name(self):
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details




class MultiTaskLoss(MultiLoss):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "MultiTaskLoss"

    def compute_loss(self, gts, preds):
        img = torch.stack([gt["img"] for gt in gts], dim=1)  # B S C H W
        B, S, _, H, W = img.shape
        background = gts[0]["background"][0]

        dataset = gts[0]['dataset'][0]
        label = gts[0]['label'][0]
        assign = preds[0]["assign"]

        if dataset == "KubricMotion":

            entropy_loss = torch.tensor(0.0).to('cuda')
            assign_bg = assign[background]
            entropy_loss_bg = - 0.1 * torch.sum(assign_bg * torch.log(assign_bg + 1e-8), dim=-1).mean()
            assign_fg = assign[~background]
            entropy_loss_fg = - 0.1 * torch.sum(assign_fg * torch.log(assign_fg + 1e-8), dim=-1).mean()

            pred_wp_track = torch.stack([pred["wp_track"] for pred in preds], dim=0)
            gt_track3d = torch.stack([gt["track3d_disp"] for gt in gts], dim=1)[0] + torch.stack([gt["world_pt_518"] for gt in gts], dim=1)[0][:1]
            
            gt_track3d_disp_fg = gt_track3d[:, ~background]
            pred_track3d_disp_fg = pred_wp_track[:, ~background]
            loss_motion = 2000 * F.mse_loss(gt_track3d_disp_fg - gt_track3d_disp_fg[:1], pred_track3d_disp_fg - pred_track3d_disp_fg[:1])
            pred_track3d_disp_bg = pred_wp_track[:, background]
            loss_background = 2000 * F.mse_loss(pred_track3d_disp_bg-pred_track3d_disp_bg[:1], torch.zeros_like(pred_track3d_disp_bg).to('cuda'))

            total = (
                loss_motion + loss_background +
                (entropy_loss_fg + entropy_loss_bg)
            ) * 10

        else:

            loss_motion = torch.tensor(0.0).to('cuda')
            loss_background = torch.tensor(0.0).to('cuda')
            loss_bg = torch.tensor(0.0).to('cuda')
            entropy_loss_fg = torch.tensor(0.0).to('cuda')
            entropy_loss_bg = torch.tensor(0.0).to('cuda')

            pred_wp_track = torch.stack([pred["wp_track"] for pred in preds], dim=0)
            gt_track3d = torch.stack([gt["track3d_disp"] for gt in gts], dim=1)[0] + torch.stack([gt["world_pt_518"] for gt in gts], dim=1)[0][:1]
            
            pred_track3d_disp_fg = pred_wp_track[:, ~background]
            gt_track3d_disp_fg = gt_track3d[:, ~background]
            loss_motion += 2000 * (F.mse_loss(gt_track3d_disp_fg - gt_track3d_disp_fg[:1], pred_track3d_disp_fg - pred_track3d_disp_fg[:1]))

            pred_track3d_disp_bg = pred_wp_track[:, background]
            loss_background = 2000 * F.mse_loss(pred_track3d_disp_bg-pred_track3d_disp_bg[:1], torch.zeros_like(pred_track3d_disp_bg).to('cuda'))

            entropy_loss = torch.tensor(0.0).to('cuda')
            assign_bg = assign[background]
            entropy_bg = -torch.sum(assign_bg * torch.log(assign_bg + 1e-8), dim=-1)
            entropy_loss_bg = 0.1 * entropy_bg.mean()
            assign_fg = assign[~background]
            entropy_fg = -torch.sum(assign_fg * torch.log(assign_fg + 1e-8), dim=-1)
            entropy_loss_fg = 0.1 * entropy_fg.mean()

            total = (
                loss_motion + loss_background + loss_bg 
                + (
                    entropy_loss_fg + entropy_loss_bg
                )
            )
            

        details = {
            "PT": float(loss_motion.detach()),
            "PTBG": float(loss_background.detach()),
            'BG': float(loss_bg.detach()),
            'ENT_FG': float(entropy_loss_fg.detach()),
            'ENT_BG': float(entropy_loss_bg.detach()),
            'total': float(total.detach())
        }

        return total, details
