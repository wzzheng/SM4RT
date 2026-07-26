
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from einops import rearrange
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from PIL import Image, ImageDraw
import random


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


class VKITTI2(BaseMultiViewDataset):
    """Motion reader that returns numpy float32 [0,1] images + per-frame tracks/vis, with custom __getitem__."""
    def __init__(self, *args, ROOT: str, **kwargs):
        self.ROOT = ROOT
        self.dataset_label = "VKITTI2"
        super().__init__(*args, **kwargs)

        split_dir = self.ROOT

        self.scenes: List[str] = []
        self.images: List[str] = []              # "<scene>/<file>"
        self.scene_img_list: List[List[int]] = []
        self.start_img_ids: List[int] = []
        self.sceneids: List[int] = []

        offset = 0
        scene_id = 0

        list_data = [dir for dir in sorted(os.listdir(split_dir)) if 'Scene' in dir]
        seq_cnt = 0
        for seq in list_data:
            if seq_cnt % 50 == 0:
                print(seq)
            seq_cnt += 1
            frames_dir = osp.join(split_dir, seq, "rgbs")
            frame_files = [f for f in sorted(os.listdir(frames_dir)) if f.lower().endswith((".jpg"))]
            num_imgs = len(frame_files)

            cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            if num_imgs < cut_off:
                continue

            ids = list(np.arange(num_imgs) + offset)
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
        # feat_idx = index[1]
        # num_views = int(index[2])
        index0 = index[0]

        W, H = getattr(self, "_resolutions", None)[0]

        start_id = self.start_img_ids[index0]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]

        pos, _ = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, np.random.default_rng(),
            min_interval=1, max_interval=1,
            video_prob=1.0, fix_interval_prob=1.0, block_shuffle=None,
        )
        # interval = 1
        # pos = sorted([random.randint(0, (len(np.array(all_image_ids)) - 1 - (num_views - 1) * interval) // 1) + i * interval for i in range(num_views)])
        img_idxs_global = np.array(all_image_ids)[pos]
        img_idxs_local = img_idxs_global - self.scene_img_list[scene_id][0]

        img_list_selected = [self.images[i] for i in img_idxs_global]
        # print(img_list_selected)

        extrinsic = np.load(osp.join(self.ROOT, self.scenes[scene_id], "extrinsics.npy"))[img_idxs_local]
        intrinsic = np.load(osp.join(self.ROOT, self.scenes[scene_id], "intrinsics.npy"))[img_idxs_local]
        visibilities = np.load(osp.join(self.ROOT, self.scenes[scene_id], "visibilities.npy"))[img_idxs_local]
        trajs_3d = np.load(osp.join(self.ROOT, self.scenes[scene_id], "trajs_3d.npy"))[img_idxs_local]
        # print(extrinsic.shape, intrinsic.shape)
        # print(visibilities.shape, trajs_3d.shape)
        # exit()

        valid_data = (trajs_3d[...,0] != 0) & (trajs_3d[...,1] != 0) & (trajs_3d[...,2] != 0)
        # (24, 35645, 3)
        # valid_data = (~np.isnan(trajs_3d)) & (trajs_3d != 0)
        # mask_nonnan = valid_data.all(axis=(0, 2))
        # print(mask_nonnan.shape)
        # print(trajs_3d.shape)
        # trajs_3d = trajs_3d[:,mask_nonnan]
        # visibilities = visibilities[:,mask_nonnan]
        # print(trajs_3d.shape)

        tracks3d_homogeneous = np.concatenate([trajs_3d, np.ones((*trajs_3d.shape[:-1], 1))], axis=-1)
        tracks3d_world = np.matmul(tracks3d_homogeneous, extrinsic[0].T)[..., :3]
        tracks3d_cam = np.matmul(tracks3d_homogeneous, extrinsic.transpose(0, 2, 1))[..., :3]
        # print(tracks3d_world.shape)
        # exit()

        scale = _scene_scale_from_track3d(torch.tensor(tracks3d_world).unsqueeze(0)).numpy()
        tracks3d_world /= scale
        tracks3d_cam /= scale
        tracks3d_reproj = project_3d2d(torch.tensor(tracks3d_world).unsqueeze(0), torch.tensor(intrinsic))[0].numpy()
        tracks3d_reproj_uv, tracks3d_reproj_d = tracks3d_reproj[...,:2], tracks3d_reproj[...,2]
        img = cv2.cvtColor(cv2.imread(osp.join(self.ROOT, self.scenes[scene_id], "rgbs", img_list_selected[0].split(os.sep, 1)[1]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        left = int((w - h) / 2)
        # print(h, w, left)
        mask = (tracks3d_reproj_uv[0][..., 0] < (w-left)) & (tracks3d_reproj_uv[0][..., 0] > left) & (tracks3d_reproj_uv[0][..., 1] < h) & (tracks3d_reproj_uv[0][..., 1] > 0)
        # exit()

        # mask = (tracks3d_reproj_uv[0][..., 0] < W) & (tracks3d_reproj_uv[0][..., 0] > 0) & (tracks3d_reproj_uv[0][..., 1] < H) & (tracks3d_reproj_uv[0][..., 1] > 0) & (visibilities[0])
        visibilities = visibilities[:,mask]
        tracks3d_cam = tracks3d_cam[:,mask]
        tracks3d_world = tracks3d_world[:,mask]
        tracks3d_reproj_uv = tracks3d_reproj_uv[:,mask]
        # track_2d = 

        valid_data = valid_data[:,mask]

        views: List[Dict[str, Any]] = []
        for i in range(self.num_views):
            rel_path = img_list_selected[i]
            scene_name, basename = rel_path.split(os.sep, 1)

            img_path = osp.join(self.ROOT, scene_name, "rgbs", osp.basename(basename))
            img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # print(img.shape, tracks3d_reproj_uv[i].shape)
            # (720, 1280, 3) (3061, 2)
            # img = cv2.imread(img_path)
            # H, W, _ = img.shape
            # coords = tracks3d_reproj_uv[i]  # (11000, 2)
            # coords = coords.astype(np.int32)
            # for coord in coords:
            #     x, y = coord
            #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            # cv2.imwrite(f'./replica_{i}exp.jpg', img)
            # exit()

            img_aug, scale, y0, x0 = _resize_center_crop(img, (H, W))

            t_k_3d_world = tracks3d_world[i]
            t_k_3d_cam = tracks3d_cam[i]
            t_k = tracks3d_reproj_uv[i]
            t_k = _augment_tracks_xy(t_k, scale, y0, x0)

            views.append(dict(
                img=ImgNorm(img_aug),
                dataset=self.dataset_label,
                label=scene_name,
                instance=osp.basename(img_path),
                track=t_k.astype(np.float32),
                track3d=t_k_3d_world.astype(np.float32),
                track3d_cam=t_k_3d_cam.astype(np.float32),
                vis=visibilities[i].astype(bool),
                valid_mask=valid_data[i].astype(bool),
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
        #     img = cv2.imread(osp.join(self.ROOT, scene_name, "rgbs", osp.basename(basename)))
        #     scene_name_i, basename_i = img_list_selected[i].split(os.sep, 1)
        #     img_i = cv2.imread(osp.join(self.ROOT, scene_name_i, "rgbs", osp.basename(basename_i)))
        #     H, W, _ = img.shape
        #     left = int((W-H) / 2)
        #     img_i = cv2.resize(img_i[:,left:W-left], (518, 518))
        #     img_rgb_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
        #     img = cv2.resize(img[:,left:W-left], (518, 518))
        #     coords = track_[i].astype(np.int32)
        #     for coord in coords:
        #         x, y = coord
        #         cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     frames_track.append(img_rgb)
        #     frames.append(img_rgb_i)

        # imageio.mimsave('./vkitti2_exp.gif', frames_track, fps=2)
        # imageio.mimsave('./vkitti2_vid.gif', frames, fps=2)
        # exit()

        return views

