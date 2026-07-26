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

from sm4rt.utils.geometry import unproject_depth_map_to_point_map


def save_depth_gif(tensor, path):
    t = tensor[0]
    t = (t - t.min()) / (t.max() - t.min()) * 255
    frames = [np.uint8(f.cpu().numpy()) for f in t]
    imageio.mimsave(path, frames, 'GIF', duration=0.2)


def save_compare(img, gt, pred, path, mask=None):
    # print(img.shape, gt.shape, pred.shape)
    img = img.detach()
    gt = gt.detach()
    pred = pred.detach()
    pred = pred.clone()
    pred[~mask] = 0
    
    img_t = img[0]  # [8, 3, 294, 518]
    img_frames = []
    for f in img_t:
        # [3, H, W] -> [H, W, 3]
        frame = f.cpu().numpy().transpose(1, 2, 0)
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        img_frames.append(frame)
    
    gt_t = gt[0]
    gt_frames = []
    for f in gt_t:
        frame = f.cpu().numpy()
        # 归一化到[0,255]
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
        frame = np.uint8(frame)
        gt_frames.append(frame)
    
    pred_t = pred[0]  # [8, 294, 518]
    pred_frames = []
    for f in pred_t:
        frame = f.cpu().numpy()
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
        frame = np.uint8(frame)
        pred_frames.append(frame)
    
    combined_frames = []
    for img_f, gt_f, pred_f in zip(img_frames, gt_frames, pred_frames):
        gt_rgb = np.stack([gt_f, gt_f, gt_f], axis=-1)
        pred_rgb = np.stack([pred_f, pred_f, pred_f], axis=-1)
        combined = np.concatenate([img_f, gt_rgb, pred_rgb], axis=1)
        combined_frames.append(combined)
    
    imageio.mimsave(path, combined_frames, 'GIF', duration=0.2)
    exit()



def plot_tracks_3d(gts, preds, save_path="track_comparison.png"):
    """
    绘制并保存 3D 轨迹对比图
    :param gts: Ground Truth 轨迹, shape (N, 3)
    :param preds: Predicted 轨迹, shape (N, 3)
    :param save_path: 图片保存路径
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取 x, y, z 坐标
    gt_x, gt_y, gt_z = gts[:, 0], gts[:, 1], gts[:, 2]
    pred_x, pred_y, pred_z = preds[:, 0], preds[:, 1], preds[:, 2]

    # 绘制 Ground Truth (通常为红色或蓝色实线)
    ax.plot(gt_x, gt_y, gt_z, label='Ground Truth', color='blue', linewidth=2, linestyle='-')
    # 标记起点和终点
    ax.scatter(gt_x[0], gt_y[0], gt_z[0], c='blue', marker='o', s=50)
    # ax.text(gt_x[0], gt_y[0], gt_z[0], ' Start', color='blue')

    # 绘制 Prediction (通常为绿色或橙色虚线/实线)
    ax.plot(pred_x, pred_y, pred_z, label='Prediction', color='orange', linewidth=2, linestyle='-')
    ax.scatter(pred_x[0], pred_y[0], pred_z[0], c='orange', marker='o', s=50)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Trajectory Comparison (GT vs Pred)')
    ax.legend()

    margin = 0.05 
    ax.margins(margin) 
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    # print(f"Image saved to {save_path}")
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=150)
    # print(f"Image saved to {save_path}")


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


class DepthScaleShiftInvLoss(BaseCriterion):
    """scale and shift invariant loss"""

    def __init__(self, reduction="mean"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"

        valid_pixel_count = mask.sum()
        if valid_pixel_count == 0:
            return (pred * 0).sum()
        
        dist = self.distance(pred, gt, mask)
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x, mask):
        x_valid = x[mask]
        splits = mask.sum(dim=(1, 2)).tolist()
        x_valid_list = torch.split(x_valid, splits)
        shift = [x.mean() for x in x_valid_list]
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]
        scale = [x.abs().mean() for x in x_valid_centered]
        scale = torch.stack(scale)
        shift = torch.stack(shift)
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(self, pred, gt, mask):
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)
        return torch.abs((pred - gt)[mask])


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    # print(prediction.shape, target.shape, mask.shape)
    # exit()
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normal
    # ize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        # return 0
        return prediction.new_tensor(0.0)
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss



def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = gradient_loss, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total



from sm4rt.utils.transform import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri, affine_inverse, as_homogeneous

def compute_median_scale(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_diff = pred[1:] - pred[:-1]  # (N-1, 3)
    gt_diff = gt[1:] - gt[:-1]        # (N-1, 3)
    
    pred_seg_lengths = torch.norm(pred_diff, dim=1)  # (N-1,)
    gt_seg_lengths = torch.norm(gt_diff, dim=1)      # (N-1,)
    
    # 3. 计算逐段比例，并取中位数
    safe_pred_lengths = pred_seg_lengths.clamp(min=1e-8)
    local_scales = gt_seg_lengths / safe_pred_lengths
    
    # 使用 torch.median 获取中位数，完美剔除异常跳变
    median_scale = torch.median(local_scales).item()
    
    return median_scale


class MultiTaskLoss(MultiLoss):
    def __init__(self):
        super().__init__()
        self.depth_loss = DepthScaleShiftInvLoss()

    def get_name(self):
        return "MultiTaskLoss"

    def compute_loss(self, gts, preds):
        img = torch.cat([gt['img'] for gt in gts], dim=0).unsqueeze(0)
        preds_pose_enc = preds["pose_enc"]
        preds_depth = preds["depth"]
        preds_depth_conf = preds["depth_conf"]
        B, N, H, W = preds_depth.shape

        pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(preds_pose_enc, (H, W))
        wp, _ = unproject_depth_map_to_point_map(preds_depth[0][..., None], pred_extrinsics[0],  pred_intrinsics[0])
        euclidean_dist = torch.norm(wp, p=2, dim=-1)
        scale = euclidean_dist.mean()
        scale_loss = (torch.log(scale.clamp(min=1e-6)) - torch.log(torch.tensor(1.0))) ** 2

        gts_intrinsics = torch.stack([gt["intrinsic"] for gt in gts], dim=1)
        gts_extrinsics = torch.stack([gt["extrinsic"] for gt in gts], dim=1)

        inv_pose_0 = torch.linalg.inv(gts_extrinsics[0,0])  # (4, 4)
        gts_extrinsics = (inv_pose_0 @ gts_extrinsics[0]).unsqueeze(0)  # (N, 4, 4)

        gts_pose_enc = extri_intri_to_pose_encoding(extrinsics=(gts_extrinsics), intrinsics=gts_intrinsics, image_size_hw=(H, W),)

        gts_pose_enc_T   =   gts_pose_enc[0,:,:3] - gts_pose_enc[0,:1,:3]
        preds_pose_enc_T = preds_pose_enc[0,:,:3]
        scale = compute_median_scale(preds_pose_enc_T, gts_pose_enc_T)
        preds_pose_enc_T *= scale

        gts_pose_enc_R   =   gts_pose_enc[0,:,3:7]
        preds_pose_enc_R = preds_pose_enc[0,:,3:7]

        gts_pose_enc_fov   =   gts_pose_enc[0,:,7:]
        preds_pose_enc_fov = preds_pose_enc[0,:,7:]

        loss_camera_T = (gts_pose_enc_T - preds_pose_enc_T).abs().mean()
        loss_camera_R = (gts_pose_enc_R - preds_pose_enc_R).abs().mean()
        loss_camera_fov = (gts_pose_enc_fov - preds_pose_enc_fov).abs().mean()
        loss_camera = 5 * (loss_camera_T + loss_camera_R + loss_camera_fov + scale_loss)

        dataset = gts[0]['dataset'][0]
        total = loss_camera

        details = {
            "loss_camera": float(loss_camera.detach()),
            "loss_T": float(loss_camera_T.detach()),
            "loss_R": float(loss_camera_R.detach()),
            "loss_FL": float(loss_camera_fov.detach()),
            'scale': float(scale_loss.detach()),
            'total': float(total.detach())
        }

        return total, details
