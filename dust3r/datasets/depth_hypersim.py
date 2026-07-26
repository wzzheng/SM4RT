
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


class HyperSIMDepth(BaseMultiViewDataset):
    """Motion reader that returns numpy float32 [0,1] images + per-frame tracks/vis, with custom __getitem__."""
    def __init__(self, *args, ROOT: str, **kwargs):
        self.ROOT = ROOT
        self.dataset_label = "HyperSIMDepth"
        super().__init__(*args, **kwargs)

        split_dir = self.ROOT

        self.scenes: List[str] = []
        self.images: List[str] = []              # "<scene>/<file>"
        self.depths: List[str] = []              # "<scene>/<file>"
        self.cameras: List[str] = []              # "<scene>/<file>"
        self.scene_img_list: List[List[int]] = []
        self.start_img_ids: List[int] = []
        self.sceneids: List[int] = []

        offset = 0
        scene_id = 0

        list_data = []
        for dir in sorted(os.listdir(split_dir)):
            for subdir in sorted(os.listdir(os.path.join(self.ROOT, dir))):
                list_data.append(f"{self.ROOT}/{dir}/{subdir}")
                # print(f"{self.ROOT}/{dir}/{subdir}")
        seq_cnt = 0

        for seq in list_data:
            seq_cnt += 1
            frame_files  = [f for f in sorted(os.listdir(seq)) if f.lower().endswith(("rgb.png"))]
            depth_files  = [f for f in sorted(os.listdir(seq)) if f.lower().endswith(("depth.npy"))]
            camera_files = [f for f in sorted(os.listdir(seq)) if f.lower().endswith(("cam.npz"))]
            num_imgs = len(frame_files)
            # cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            # print(seq, len(frame_files), cut_off, not self.allow_repeat)
            # if num_imgs < cut_off:
            ids = list(np.arange(num_imgs) + offset)
            self.scene_img_list.append(ids)
            self.scenes.append(seq)
            self.images.extend([osp.join(seq, ff) for ff in frame_files])
            self.depths.extend([osp.join(seq, ff) for ff in depth_files])
            self.cameras.extend([osp.join(seq, ff) for ff in camera_files])
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

        # print(num_views, start_id, all_image_ids)
        pos, _ = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, np.random.default_rng(),
            min_interval=8, max_interval=8,
            video_prob=1.0, fix_interval_prob=1.0, block_shuffle=None,
        )
        img_idxs_global = np.array(all_image_ids)[pos]
        img_idxs_local = img_idxs_global - self.scene_img_list[scene_id][0]

        img_list_selected =   [self.images[i] for i in img_idxs_global]
        depth_list_selected = [self.depths[i] for i in img_idxs_global]
        cam_list_selected =   [self.cameras[i] for i in img_idxs_global]

        views: List[Dict[str, Any]] = []
        for i in range(self.num_views):
            img_path = img_list_selected[i]

            scene_name = str(Path(*Path(img_path).parts[-4:-1]))
            image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            depth = np.load(depth_list_selected[i])
            depth = np.nan_to_num(depth, nan=0.0)

            sky_mask = depth >= 1000
            depth[sky_mask] = -1.0  # sky
            depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
            threshold = (np.percentile(depth[depth > 0], 98) if depth[depth > 0].size > 0 else 0)
            depth[depth > threshold] = 0.0

            camera = np.load(cam_list_selected[i], allow_pickle=True)
            # print(camera.keys())
            # exit()
            extrinsic = camera["pose"]
            intrinsic = camera["intrinsics"]
            # print(extrinsic.shape, intrinsic.shape)
            # exit()

            rng = np.random.default_rng(seed=42)
            image, depth, intrinsic = self._crop_resize_if_necessary(
                image, depth, intrinsic, (W, H), rng=rng, info=None
            )
            image = np.array(image).astype(np.float32) / 255.0

            # print(np.min(image), np.max(image), image.shape, depth.shape, intrinsic.shape)

            views.append(dict(
                img=ImgNorm(image),
                depth=depth,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                dataset=self.dataset_label,
                label=scene_name,
                instance=osp.basename(img_path),
                reproj=True,
                motion=True,
                is_metric=False,
            ))

        return views

