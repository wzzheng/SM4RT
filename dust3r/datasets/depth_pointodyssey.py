
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
from pathlib import Path

import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor()])



def _resize_center_crop(img: np.ndarray, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    th, tw = target_hw  # (H, W)
    H, W = img.shape[:2]
    if th <= 0 or tw <= 0:
        return img, 1.0, 0, 0
    scale = max(th / max(H, 1), tw / max(W, 1))
    newH = int(round(H * scale))
    newW = int(round(W * scale))
    if newH != H or newW != W:
        img_r = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
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


def resize_intrinsics(fx_, fy_, w_, h_, new_wh=(518, 294)):
    new_w, new_h = new_wh
    
    orig_w = w_ * 2
    orig_h = h_ * 2
    
    sx = new_w / orig_w
    sy = new_h / orig_h
    
    fx = fx_ * sx  # fx
    fy = fy_ * sy  # fx
    return fx, fy


class PointOdyssey(BaseMultiViewDataset):
    """Motion reader that returns numpy float32 [0,1] images + per-frame tracks/vis, with custom __getitem__."""
    def __init__(self, *args, ROOT: str, **kwargs):
        self.ROOT = ROOT
        self.dataset_label = "PointOdyssey"
        super().__init__(*args, **kwargs)

        split_dir = self.ROOT

        self.scenes: List[str] = []
        self.images: List[str] = []              # "<scene>/<file>"
        self.cameras: List[str] = []              # "<scene>/<file>"
        self.scene_img_list: List[List[int]] = []
        self.start_img_ids: List[int] = []
        self.sceneids: List[int] = []

        offset = 0
        scene_id = 0

        list_data = []
        for dir in sorted(os.listdir(split_dir)):
            if ('.mp4' not in dir) and ('.py' not in dir):
                list_data.append(f"{self.ROOT}/{dir}/")
                # print(f"{self.ROOT}/{dir}/")
        seq_cnt = 0

        for seq in list_data:
            seq_cnt += 1
            frame_files  = [f for f in sorted(os.listdir(f"{seq}/rgbs")) if f.lower().endswith((".jpg"))]
            # camera_files = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith(("cam.npz"))]
            num_imgs = len(frame_files)
            ids = list(np.arange(num_imgs) + offset)
            self.scene_img_list.append(ids)
            self.scenes.append(seq)
            self.images.extend([osp.join(seq, 'rgbs', ff) for ff in frame_files])
            # self.cameras.extend([osp.join(seq, ff) for ff in camera_files])
            self.start_img_ids.extend(ids[: num_imgs - self.num_views + 1])
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

        W, H = getattr(self, "_resolutions", None)[0]

        start_id = self.start_img_ids[index0]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        # print(scene_id)
        # exit()

        # print(num_views, start_id, all_image_ids)
        pos, _ = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, np.random.default_rng(),
            min_interval=8, max_interval=8,
            video_prob=1.0, fix_interval_prob=1.0, block_shuffle=None,
        )
        img_idxs_global = np.array(all_image_ids)[pos]
        img_idxs_local = img_idxs_global - self.scene_img_list[scene_id][0]

        img_list_selected =   [self.images[i] for i in img_idxs_global]

        scene_name = img_list_selected[0].split('/')[-3]
        intrinsics = np.load(f"{img_list_selected[0].split(scene_name)[0]}/{scene_name}/intrinsics.npy")[img_idxs_local]
        extrinsics = np.load(f"{img_list_selected[0].split(scene_name)[0]}/{scene_name}/extrinsics.npy")[img_idxs_local]

        views: List[Dict[str, Any]] = []
        for i in range(self.num_views):
            img_path = img_list_selected[i]
            # image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # h_, w_, _ = image.shape
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h_, w_, _ = img_bgr.shape
            img_aug = _resize_center_crop(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (H, W))

            extrinsic = extrinsics[i]
            intrinsic = intrinsics[i]
            fx, fy = resize_intrinsics(intrinsic[0,0], intrinsic[1,1], w_/2, h_/2)
            intrinsic = np.array([
                [fx, 0., W/2],
                [0., fy, H/2],
                [0., 0., 1.],
            ])

            # rng = np.random.default_rng(seed=42)
            # image, _, intrinsic = self._crop_resize_if_necessary(
            #     image, _, intrinsic, (W, H), rng=rng, info=None
            # )
            # image = np.array(image).astype(np.float32) / 255.0

            views.append(dict(
                img=ImgNorm(img_aug),
                intrinsic=intrinsic,
                extrinsic=np.linalg.inv(extrinsic),
                dataset=self.dataset_label,
                label=scene_name,
                instance=osp.basename(img_path),
                reproj=True,
                motion=True,
                is_metric=False,
            ))

        return views

