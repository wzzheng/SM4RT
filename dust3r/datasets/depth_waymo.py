
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


class WaymoDepth(BaseMultiViewDataset):
    """Motion reader that returns numpy float32 [0,1] images + per-frame tracks/vis, with custom __getitem__."""
    def __init__(self, *args, ROOT: str, **kwargs):
        self.ROOT = ROOT
        self.dataset_label = "Waymo"
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
            list_data.append(f"{self.ROOT}/{dir}/")

        seq_cnt = 0

        for seq in list_data:
            seq_cnt += 1
            # 获取所有文件
            all_frame_files = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith((".jpg"))]
            all_depth_files = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith((".exr"))]
            all_camera_files = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith((".npz"))]
            
            # 按 IDX 分组（假设格式为 {i:05d}_{IDX}.扩展名）
            sample_groups = {}
            for f in all_frame_files:
                # 提取 IDX（下划线后面的部分，去掉扩展名）
                base_name = os.path.splitext(f)[0]
                if '_' in base_name:
                    idx = base_name.split('_')[-1]  # 获取 IDX 部分
                    if idx not in sample_groups:
                        sample_groups[idx] = {'frames': [], 'depths': [], 'cameras': []}
                    sample_groups[idx]['frames'].append(f)
            
            # 同样处理 depth 和 camera 文件
            for f in all_depth_files:
                base_name = os.path.splitext(f)[0]
                if '_' in base_name:
                    idx = base_name.split('_')[-1]
                    if idx in sample_groups:
                        sample_groups[idx]['depths'].append(f)
            
            for f in all_camera_files:
                base_name = os.path.splitext(f)[0]
                if '_' in base_name:
                    idx = base_name.split('_')[-1]
                    if idx in sample_groups:
                        sample_groups[idx]['cameras'].append(f)
            
            # 遍历每个样本 IDX
            for idx, group in sample_groups.items():
                # 确保每个样本都有对应的帧、深度和相机文件
                frame_files = sorted(group['frames'])
                depth_files = sorted(group['depths'])
                camera_files = sorted(group['cameras'])
                
                num_imgs = len(frame_files)
                
                # 如果某个样本没有对应的深度或相机文件，可以跳过或警告
                if len(depth_files) != num_imgs or len(camera_files) != num_imgs:
                    print(f"Warning: Sample {seq} IDX {idx} has mismatched files: frames={num_imgs}, depths={len(depth_files)}, cameras={len(camera_files)}")
                    continue
                
                ids = list(np.arange(num_imgs) + offset)
                self.scene_img_list.append(ids)  # 这里 scene_img_list 现在对应每个样本
                self.scenes.append(f"{seq}_{idx}")  # 场景标识包含样本 IDX
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
            min_interval=2, max_interval=2,
            video_prob=1.0, fix_interval_prob=1.0, block_shuffle=None,
        )
        img_idxs_global = np.array(all_image_ids)[pos]
        img_idxs_local = img_idxs_global - self.scene_img_list[scene_id][0]

        img_list_selected =   [self.images[i] for i in img_idxs_global]
        depth_list_selected = [self.depths[i] for i in img_idxs_global]
        cam_list_selected =   [self.cameras[i] for i in img_idxs_global]
        # print(img_list_selected[0])
        # scene_dir = os.path.dirname(img_list_selected[0])
        scene_dir = os.path.dirname(os.path.dirname(img_list_selected[0]))
        # data = np.load(osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True)
        # intrinsics = data["intrinsics"][img_idxs_local]
        # extrinsics = data["trajectories"][img_idxs_local]
        # print(intrins.shape)
        # print(trajectories.shape)


        views: List[Dict[str, Any]] = []
        for i in range(self.num_views):
            img_path = img_list_selected[i]
            depth_path = depth_list_selected[i]
            cam_path = cam_list_selected[i]

            scene_name = str(Path(*Path(img_path).parts[-4:-1]))
            image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # print(image.shape)

            # depth = imread_cv2(osp.join(depth_list_selected[i]), cv2.IMREAD_UNCHANGED)
            # depth = depth.astype(np.float32) / 1000.0
            # depth[~np.isfinite(depth)] = 0  # invalid
            
            depth = imread_cv2(depth_path)
            cam_data = np.load(cam_path, allow_pickle=True)

            extrinsic = cam_data['cam2world']
            intrinsic = cam_data['intrinsics']
            distortion = cam_data['distortion']
            # extrinsic = np.linalg.inv(extrinsic)
            # print(extrinsic.shape, intrinsic.shape)
            # print(distortion)
            # exit()

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
            # image = np.array(image).astype(np.float32) / 255.0

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



        # views: List[Dict[str, Any]] = []
        # for i in range(self.num_views):
        #     img_path = img_list_selected[i]

        #     scene_name = str(Path(*Path(img_path).parts[-4:-1]))
        #     image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        #     depth = np.load(depth_list_selected[i])
        #     depth = np.nan_to_num(depth, nan=0.0)

        #     sky_mask = depth >= 1000
        #     depth[sky_mask] = -1.0  # sky
        #     depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
        #     threshold = (np.percentile(depth[depth > 0], 98) if depth[depth > 0].size > 0 else 0)
        #     depth[depth > threshold] = 0.0

        #     camera = np.load(cam_list_selected[i], allow_pickle=True)
        #     # print(camera.keys())
        #     # exit()
        #     extrinsic = trajectories[i]
        #     intrinsic = intrins[i]
        #     # print(extrinsic.shape, intrinsic.shape)
        #     # exit()

        #     rng = np.random.default_rng(seed=42)
        #     image, depth, intrinsic = self._crop_resize_if_necessary(
        #         image, depth, intrinsic, (W, H), rng=rng, info=None
        #     )
        #     image = np.array(image).astype(np.float32) / 255.0

        #     # print(np.min(image), np.max(image), image.shape, depth.shape, intrinsic.shape)

        #     views.append(dict(
        #         img=ImgNorm(image),
        #         depth=depth,
        #         intrinsic=intrinsic,
        #         extrinsic=extrinsic,
        #         dataset=self.dataset_label,
        #         label=scene_name,
        #         instance=osp.basename(img_path),
        #         reproj=True,
        #         motion=True,
        #         is_metric=False,
        #     ))

        # return views

