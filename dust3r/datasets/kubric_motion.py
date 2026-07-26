import time
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import imageio
from PIL import ImageDraw, Image
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
import torchlie as lie
from torchlie import SE3
from scipy.spatial.transform import Rotation
import PIL
import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor()])
import logging
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)


def _resize_center_crop(img: np.ndarray, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    th, tw = target_hw  # (H, W)
    H, W = img.shape[:2]
    if th <= 0 or tw <= 0:
        return img, 1.0, 0, 0
    scale = max(th / max(H, 1), tw / max(W, 1))
    newH = int(round(H * scale))
    newW = int(round(W * scale))
    if newH != H or newW != W:
        img_r = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST_EXACT)
    else:
        img_r = img
    y0 = max(0, (newH - th) // 2)
    x0 = max(0, (newW - tw) // 2)
    img_c = img_r[y0:y0 + th, x0:x0 + tw]
    if img_c.shape[0] != th or img_c.shape[1] != tw:
        pad = np.zeros((th, tw, img_c.shape[2]), dtype=img_c.dtype)
        pad[:img_c.shape[0], :img_c.shape[1]] = img_c
        img_c = pad
    return img_c


def save_pca_images(twist):
    """
    将形状为 [B, H, W, 6] 的张量拆分为前3通道和后3通道，
    分别进行最大值归一化并保存为 RGB 图片 (GIF)。
    
    参数:
        twist: torch.Tensor, 形状 [B, H, W, 6]
    """
    
    if twist.dim() != 4 or twist.shape[3] != 6:
        raise ValueError(f"输入张量形状应为 [B, H, W, 6], 当前为 {twist.shape}")

    B, H, W, C = twist.shape
    data_np = twist.detach().cpu().numpy()
    
    def normalize_and_convert(data_part):
        normalized_data = np.zeros_like(data_part, dtype=np.float32)
        for i in range(3):
            channel = data_part[..., i]
            min_val = channel.min()
            max_val = channel.max()
            
            if max_val - min_val > 1e-8:
                channel_norm = (channel - min_val) / (max_val - min_val) * 255.0
            else:
                channel_norm = np.zeros_like(channel)
            
            normalized_data[..., i] = channel_norm
            
        return np.clip(normalized_data, 0, 255).astype(np.uint8)

    data_v = data_np[:, :, :, :3]
    data_w = data_np[:, :, :, 3:]
    
    data_image_v = normalize_and_convert(data_v)
    data_image_w = normalize_and_convert(data_w)
    
    frames_v = []
    for i in range(B):
        img_array = data_image_v[i]
        img = Image.fromarray(img_array, mode='RGB')
        frames_v.append(np.array(img))
    path_v = 'twist_v.gif'
    imageio.mimwrite(path_v, frames_v, format='GIF', duration=100, loop=0)
    
    frames_w = []
    for i in range(B):
        img_array = data_image_w[i]
        img = Image.fromarray(img_array, mode='RGB')
        frames_w.append(np.array(img))
    path_w = 'twist_w.gif'
    imageio.mimwrite(path_w, frames_w, format='GIF', duration=100, loop=0)



@torch.no_grad()
def _scene_scale_from_track3d(track3d_bsn3: torch.Tensor,
                              visible_bsn: torch.Tensor | None = None) -> torch.Tensor:
    assert track3d_bsn3.dim() == 4 and track3d_bsn3.size(-1) == 3, "track3d 需为 [B,S,N,3]"
    d_bsn = torch.linalg.norm(track3d_bsn3.float(), dim=-1)  # (B,S,N)
    scales = []
    if visible_bsn is not None:
        m = visible_bsn[0].bool()
        vals = d_bsn[0][m]
    else:
        vals = d_bsn[0].reshape(-1)

    if vals.numel() == 0:
        scales.append(torch.tensor(1.0, device=track3d_bsn3.device, dtype=track3d_bsn3.dtype))  # 回退
    else:
        scales.append(vals.mean().to(dtype=track3d_bsn3.dtype))  # 与 FinetuneLoss 中“masked 平均”一致
    return torch.stack(scales, dim=0)  # (B,)


class Kubric_Motion(BaseMultiViewDataset):
    """
    Kubric motion dataset that follows the reading/processing in kubric.py,
    adapted to the BaseMultiViewDataset triplet indexing and resolutions.
    It loads dense annotations with keys: coords (H*W, T, 2), visibility (H*W, T).
    For each frame, it outputs:
      img: float32 [0,1] after resize+center-crop to the selected (W,H)
      track: (N,2) float32 in pixel coordinates after the same transform
      vis: (N,) bool where True means occluded (inverted from visibility)
    """

    def __init__(
        self,
        *args,
        ROOT: str,
        zfill: int = 3,
        max_sequences: int = 6000,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.dataset_label = "KubricMotion"
        self.zfill = zfill
        self.max_sequences = max_sequences
        super().__init__(*args, **kwargs)

        self.store: Dict[str, Dict[str, Any]] = {}

        seq_list = sorted(os.listdir(self.ROOT))
        seq_cnt = 0
        for seq in seq_list:
            # if seq_cnt % 100 == 0:
            #     print(seq)
            seq_cnt += 1
            if seq_cnt > self.max_sequences:
                break
            frame_path    = osp.join(self.ROOT, seq, "frames")
            seg_path      = osp.join(self.ROOT, seq, "segmentations", '000.png')
            world_pt_path = osp.join(self.ROOT, seq, f"{seq}_world_pts.pt")
            twist_path = osp.join(self.ROOT, seq, f"{seq}_twist.pt")
            frames = [osp.join(frame_path, f"{i:03d}.png") for i in range(self.num_views)]
            scale_path = osp.join(self.ROOT, seq, f"{seq}_scale.txt")
            dense_path = osp.join(self.ROOT, seq, f"{seq}_dense.npy")

            self.store[seq] = {
                "frames": frames, 
                "seg_path": seg_path, 
                "world_pt_path": world_pt_path,
                "twist_path": twist_path,
                "scale_path": scale_path,
                "dense_path": dense_path,
            }

        print("[KUBRIC] Sample Length:", seq_cnt)
        self.sequence_list = list(self.store.keys())
        self.cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        self.all_ref = []
        for seq in self.sequence_list:
            T = len(self.store[seq]["frames"])
            if T >= self.cut_off:
                self.all_ref.extend([(seq, 0)])

        self.invalid_seq = {seq: False for seq in self.sequence_list}

    def __len__(self) -> int:
        return len(self.all_ref)

    def __getitem__(self, index: Any):
        global torch
        if isinstance(index, (tuple, list)):
            index, ar_idx, n_view = index
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            n_view = self.num_views

        try:
            seed = torch.randint(0, 2**32, (1,)).item()
        except Exception:
            seed = int(time.time() * 1e6) & 0xFFFFFFFF
        rng = np.random.default_rng(seed=seed)
        self._rng = rng
        seq_name, start_id = self.all_ref[index]
        frames = self.store[seq_name]["frames"]
        pos = list(range(len(frames)))[:self.num_views]

        world_pt = torch.load(self.store[seq_name]["world_pt_path"], map_location='cpu', weights_only=False)[0,pos]
        world_pt_518 = torch.nn.functional.interpolate(world_pt.permute(0, 3, 1, 2), size=(518, 518), mode='nearest-exact').permute(0, 2, 3, 1).float()

        scale = float(open(self.store[seq_name]["scale_path"]).read().strip())
        Frame, H, W, _ = world_pt_518.shape

        track3d_disp_diff = world_pt_518 - world_pt_518[:1]
        diff = track3d_disp_diff[1:] - track3d_disp_diff[:-1]
        frame_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        norm = frame_norms.sum(dim=0)

        euclidean_dist = torch.norm(world_pt_518, p=2, dim=-1)
        scale_ = euclidean_dist.mean()
        world_pt_518 /= scale
        track3d_disp = world_pt_518 - world_pt_518[:1]

        seg_path = self.store[seq_name]["seg_path"]
        img = Image.open(seg_path).convert('L')
        img_np = np.array(img)

        seg_mask = torch.from_numpy(img_np).long()
        seg_mask_518 = torch.nn.functional.interpolate(seg_mask.unsqueeze(0).unsqueeze(0).float(), size=(518, 518), mode='nearest-exact', align_corners=None)[0,0].long()
        background = (seg_mask_518 == 0) | (norm < 0.01)

        views: List[Dict[str, Any]] = []

        # if 'val' not in self.ROOT:
        #     pos = [0,2,4,6,8,10,12,14,16,18,20,22]

        # pos = [0,2,4,6,8,10,12,14,16,18,20,22]
        pos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        for s_idx, p in enumerate(pos):
            img_path = frames[p]
            img = PIL.ImageOps.exif_transpose(PIL.Image.open(img_path)).convert("RGB")
            img = img.resize((518, 518), PIL.Image.LANCZOS)
            views.append(dict(
                img=ImgNorm(img),
                dataset=self.dataset_label,
                label=seq_name,
                instance=osp.basename(img_path),
                track3d_disp=track3d_disp[p],
                world_pt_518=world_pt_518[p],
                # twist=twist[p],
                background=background,
                seg_mask=seg_mask_518,
                is_metric=False,
            ))
        return views

