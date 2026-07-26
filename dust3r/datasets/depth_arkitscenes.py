
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


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ARKitScenesDepth(BaseMultiViewDataset):
    """Motion reader that returns numpy float32 [0,1] images + per-frame tracks/vis, with custom __getitem__."""
    def __init__(self, *args, ROOT: str, **kwargs):
        self.ROOT = f"{ROOT}/Training"
        self.dataset_label = "ARKitScenesDepth"
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
        # list_ = []
        for dir in sorted(os.listdir(split_dir)):
            if '.' not in dir:
                # print(dir)
                list_data.append(f"{self.ROOT}/{dir}/")
                # list_.append(dir)

        seq_cnt = 0
        for seq in list_data:
            seq_cnt += 1
            frame_files  = [f for f in sorted(os.listdir(f"{seq}/vga_wide")) if f.lower().endswith((".jpg"))]
            depth_files  = [f for f in sorted(os.listdir(f"{seq}/lowres_depth")) if f.lower().endswith((".png"))]
            num_imgs = len(frame_files)
            # cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            # print(seq, len(frame_files), cut_off, not self.allow_repeat)
            # if num_imgs < cut_off:
            ids = list(np.arange(num_imgs) + offset)
            self.scene_img_list.append(ids)
            self.scenes.append(seq)
            self.images.extend([osp.join(f"{seq}/vga_wide", ff) for ff in frame_files])
            self.depths.extend([osp.join(f"{seq}/lowres_depth", ff) for ff in depth_files])
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
        # cam_list_selected =   img_idxs_local
        # print(img_list_selected[0])
        scene_dir = os.path.dirname(os.path.dirname(img_list_selected[0]))
        # print(scene_dir)
        # exit()

        data = np.load(osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True)
        intrins = data["intrinsics"][img_idxs_local]
        trajectories = data["trajectories"][img_idxs_local]
        K = np.expand_dims(np.eye(3), 0).repeat(self.num_views, 0)
        K[:, 0, 0] = [fx for _, _, fx, _, _, _ in intrins]
        K[:, 1, 1] = [fy for _, _, _, fy, _, _ in intrins]
        K[:, 0, 2] = [cx for _, _, _, _, cx, _ in intrins]
        K[:, 1, 2] = [cy for _, _, _, _, _, cy in intrins]
        extrinsics = trajectories
        intrinsics = K

        views: List[Dict[str, Any]] = []
        for i in range(self.num_views):
            img_path = img_list_selected[i]
            extrinsic = extrinsics[i]
            intrinsic = intrinsics[i]


            scene_name = str(Path(*Path(img_path).parts[-4:-1]))
            image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            depth = imread_cv2(osp.join(depth_list_selected[i]), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0  # invalid


            # depth = np.load(depth_list_selected[i])
            # depth[~np.isfinite(depth)] = 0  # invalid
            # threshold = (
            #     np.percentile(depth[depth > 0], 98)
            #     if depth[depth > 0].size > 0
            #     else 0
            # )
            # depth[depth > threshold] = 0.0
            # depth[depth > 1000] = 0.0
            # print(intrinsic.shape, image.shape, depth.shape)

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

