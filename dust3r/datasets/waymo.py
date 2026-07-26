
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from einops import rearrange
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from PIL import Image, ImageDraw
import json

import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor()])


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


# intrs: T, 3, 3
# trajs: 1, T, N, 3
def project_3d2d(trajs_3d, intrs):
    B, T, N = trajs_3d.shape[:3]
    trajs_3d = trajs_3d.reshape(-1, N, 3)
    intrs = intrs.reshape(-1, 3, 3)
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]
    trajs_uvd = torch.zeros((T, N, 3), device=trajs_3d.device)
    trajs_uvd[..., 0] = trajs_3d[..., 0] * fx[..., None] / trajs_3d[..., 2] + cx[..., None]
    trajs_uvd[..., 1] = trajs_3d[..., 1] * fy[..., None] / trajs_3d[..., 2] + cy[..., None]
    trajs_uvd[..., 2] = trajs_3d[..., 2]
    trajs_uvd = rearrange(trajs_uvd, "(B T) N C -> B T N C", T=T)
    return trajs_uvd

def reproject_2d3d(trajs_uvd, intrs):
    B, T, N = trajs_uvd.shape[:3]
    trajs_uvd = trajs_uvd.reshape(-1, N, 3)
    intrs = intrs.reshape(-1, 3, 3)
    B, N = trajs_uvd.shape[:2]
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]
    trajs_3d = torch.zeros((B, N, 3), device=trajs_uvd.device)
    trajs_3d[..., 0] = trajs_uvd[..., 2] * (trajs_uvd[..., 0] - cx[..., None]) / fx[..., None]
    trajs_3d[..., 1] = trajs_uvd[..., 2] * (trajs_uvd[..., 1] - cy[..., None]) / fy[..., None]
    trajs_3d[..., 2] = trajs_uvd[..., 2]
    return rearrange(trajs_3d, "(B T) N C -> B T N C", T=T)


def _resize_center_crop(img: np.ndarray, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    th, tw = target_hw  # (H, W)
    H, W = img.shape[:2]
    if th <= 0 or tw <= 0:
        return img, 1.0, 0, 0
    scale = max(th / H, tw / W)  # cover crop
    nh, nw = int(round(H * scale)), int(round(W * scale))
    if nh != H or nw != W:
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    y0 = max(0, (nh - th) // 2)
    x0 = max(0, (nw - tw) // 2)
    img = img[y0:y0+th, x0:x0+tw]
    return img, scale, y0, x0


def _augment_tracks_xy(tracks_xy: np.ndarray, scale: float, y0: int, x0: int) -> np.ndarray:
    if tracks_xy is None or tracks_xy.size == 0:
        return np.zeros_like(tracks_xy, dtype=np.float32)
    out = tracks_xy.astype(np.float32).copy()
    out[..., 0] = out[..., 0] * scale - x0  # x
    out[..., 1] = out[..., 1] * scale - y0  # y
    return out


def sample_indices(num_frame, num_views):
    rng = np.random.default_rng()
    max_val = min(num_views, num_frame)
    # max_val = min(num_views * 2, num_frame)
    rest = rng.choice(np.arange(1, max_val), size=num_views - 1, replace=False)
    return sorted(np.concatenate(([0], rest)).tolist())

class Waymo(BaseMultiViewDataset):
    """Motion reader that returns numpy float32 [0,1] images + per-frame tracks/vis, with custom __getitem__."""
    def __init__(self, *args, ROOT: str, **kwargs):
        self.ROOT = ROOT
        self.dataset_label = "PointOdyssey"
        super().__init__(*args, **kwargs)

        split = getattr(self, "split", "train")
        split_dir = osp.join(self.ROOT, split)
        if not osp.isdir(split_dir):
            split_dir = self.ROOT

        self.scenes: List[str] = []
        self.images: List[str] = []              # "<scene>/<file>"
        self.scene_img_list: List[List[int]] = []
        self.start_img_ids: List[int] = []
        self.scenes_img_num: List[int] = []
        self.sceneids: List[int] = []

        with open("/data1/DA3/AAAVISUALIZATION/SM4G/drivetrack/track/filenames.json", "r", encoding="utf-8") as f:
            filename_list = json.load(f)
        offset = 0
        scene_id = 0
        seq_cnt = 0
        for seq in filename_list:
            if seq_cnt % 50 == 0:
                print(seq)
            seq_cnt += 1
            sd = osp.join(split_dir, seq)
            frames_dir = osp.join(sd, "img")

            frame_files = [f for f in sorted(os.listdir(frames_dir)) if f.lower().endswith((".jpg"))]
            num_imgs = len(frame_files)
            img = cv2.imread(f"{frames_dir}/{frame_files[0]}")
            H, W, _ = img.shape
            
            cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            if num_imgs < cut_off:
                continue

            ids = list(np.arange(num_imgs) + offset)
            self.scenes_img_num.append(num_imgs)
            self.scene_img_list.append(ids)
            self.scenes.append(seq)
            self.images.extend([osp.join(seq, ff) for ff in frame_files])
            self.start_img_ids.extend(ids[: num_imgs - cut_off + 1])

            offset += num_imgs
            scene_id += 1
        
        for sid, img_ids in enumerate(self.scene_img_list):
            self.sceneids.extend([sid] * len(img_ids))
        assert len(self.sceneids) == len(self.images), "sceneids/images mismatch"

    def __len__(self) -> int:
        return len(self.start_img_ids)

    def __getitem__(self, index: Any):
        # Parse triplet index
        num_views = self.num_views
        index0 = index[0]

        start_id = self.start_img_ids[index0]
        scene_id = self.sceneids[start_id]
        num_frame = self.scenes_img_num[scene_id]
        pos = sample_indices(num_frame, num_views)
        all_image_ids = self.scene_img_list[scene_id]

        W, H = getattr(self, "_resolutions", None)[0]
        img_idxs_local = (np.array(all_image_ids) - self.scene_img_list[scene_id][0])[pos]
        img_list_selected = [self.images[i] for i in img_idxs_local]

        scene_name_, basename_ = img_list_selected[0].split(os.sep, 1)
        img = Image.open(osp.join(self.ROOT, scene_name_, "img", osp.basename(basename_)))
        width, height = img.size

        anno_ = np.load(osp.join(self.ROOT, scene_name_, "metadata.npz"), mmap_mode="r")
        tracks_uv  = anno_["tracks_uv"][img_idxs_local]
        tracks_xyz_cam = anno_['tracks_xyz_cam'][img_idxs_local]
        tracks_xyz_world = anno_['tracks_xyz_world'][img_idxs_local]
        visibility    = anno_["visibility"][img_idxs_local]
        intrinsics    = anno_["intrinsics"]
        # print(tracks_uv.shape, tracks_xyz_cam.shape, tracks_xyz_world.shape, visibility.shape, intrinsics.shape)
        tracks_uv_0 = tracks_uv[0]
        visibility_0 = visibility[0]

        mask_u = (0 < tracks_uv_0[..., 0]) & (tracks_uv_0[..., 0] < width)
        mask_v = (0 < tracks_uv_0[..., 1]) & (tracks_uv_0[..., 1] < height)
        mask = mask_u & mask_v & visibility_0

        tracks_uv  = tracks_uv[:,mask]
        tracks_xyz_cam = tracks_xyz_cam[:,mask]
        tracks_xyz_world = tracks_xyz_world[:,mask]

        scene_scale = _scene_scale_from_track3d(torch.tensor(tracks_xyz_world).unsqueeze(0)).numpy()
        tracks_xyz_cam /= scene_scale
        tracks_xyz_world /= scene_scale

        views: List[Dict[str, Any]] = []
        for i in range(self.num_views):
            rel_path = img_list_selected[i]
            scene_name, basename = rel_path.split(os.sep, 1)

            img_path = osp.join(self.ROOT, scene_name, "img", osp.basename(basename))
            img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img_aug, scale, y0, x0 = _resize_center_crop(img, (H, W))

            t_k_3d_world = tracks_xyz_world[i]
            t_k_3d_cam = tracks_xyz_cam[i]

            t_k = tracks_uv[i]
            t_k = _augment_tracks_xy(t_k, scale, y0, x0)

            views.append(dict(
                img=ImgNorm(img_aug),
                dataset=self.dataset_label,
                label=scene_name,
                instance=osp.basename(img_path),
                track=t_k.astype(np.float32),
                track3d=t_k_3d_world.astype(np.float32),
                track3d_cam=t_k_3d_cam.astype(np.float32),
                scale=scene_scale,
                reproj=True,
                motion=True,
                is_metric=False,
            ))

        # track_ = torch.cat([torch.tensor(view['track']).unsqueeze(0) for view in views], dim=0).numpy()
        # import imageio.v2 as imageio
        # frames_track = []
        # frames = []
        # for i in range(self.num_views):
        #     scene_name, basename = img_list_selected[0].split(os.sep, 1)
        #     scene_name_i, basename_i = img_list_selected[i].split(os.sep, 1)
        #     imgdir = osp.join(self.ROOT, getattr(self, "split", "train"), scene_name, "rgbs", osp.basename(basename))
        #     img = cv2.imread(imgdir)
        #     img_i = cv2.imread(osp.join(self.ROOT, getattr(self, "split", "train"), scene_name_i, "rgbs", osp.basename(basename_i)))
        #     H, W, _ = img.shape
        #     img = cv2.resize(img, (518, 294))
        #     img_i = cv2.resize(img_i, (518, 294))

        #     # img = img[hstart:H-hstart]
        #     # img_i = img_i[hstart:H-hstart]

        #     for coord in track_[i].astype(np.int32):
        #         x, y = coord
        #         cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     frames_track.append(img_rgb)
        #     frames.append(cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB))
        # imageio.mimsave('./vis/odyssey_exp.gif', frames_track, fps=6)
        # imageio.mimsave('./vis/odyssey_vid.gif', frames, fps=6)
        # exit()
        # print(views[0]['img'].max(), views[0]['img'].min())
        # exit()

        return views
