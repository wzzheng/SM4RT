
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


def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))
    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


def save_depth(depth_map, file_name):
    """
    将 float32 格式的深度图原封不动地保存为 KITTI 格式的 16-bit PNG
    """
    # 1. 逆向操作：乘回 256.0
    depth_uint16 = depth_map * 256.0
    
    # 2. 数据类型转换：转回 uint16 (16-bit)
    # 注意：如果数据有极小的浮点误差，可以先 np.clip 或 np.round，但通常直接 astype 即可
    depth_uint16 = depth_uint16.astype(np.uint16)
    
    # 3. 使用 PIL 保存
    img = Image.fromarray(depth_uint16)
    img.save(file_name)
    

def scale_invariant_center_crop(image, depth, target_W, target_H):
    """
    对 image 和 depth 进行 scale-invariant resize 然后 center crop
    """
    src_H, src_W = image.shape[:2]
    
    # 1. 计算 scale-invariant 的缩放比例 (取较大值以保证覆盖目标区域)
    scale = max(target_W / src_W, target_H / src_H)
    
    # 2. 计算缩放后的新尺寸
    new_W = int(round(src_W * scale))
    new_H = int(round(src_H * scale))
    
    # 3. 等比例 Resize
    # image 使用双线性插值
    resized_image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    # depth 使用最近邻插值，避免破坏深度值
    resized_depth = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
    
    # 4. Center Crop
    # 计算左上角起点
    x_start = (new_W - target_W) // 2
    y_start = (new_H - target_H) // 2
    
    cropped_image = resized_image[y_start : y_start + target_H, x_start : x_start + target_W]
    cropped_depth = resized_depth[y_start : y_start + target_H, x_start : x_start + target_W]
    
    return cropped_image, cropped_depth


class KittiDepth(BaseMultiViewDataset):
    """Motion reader that returns numpy float32 [0,1] images + per-frame tracks/vis, with custom __getitem__."""
    def __init__(self, *args, ROOT: str, **kwargs):
        self.ROOT = ROOT
        self.dataset_label = "Kitti"
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

        # list_data = []
        # for dir in sorted(os.listdir(split_dir)):
        #     list_data.append(f"{self.ROOT}/{dir}/")

        # seq_cnt = 0

        # for seq in list_data:
        #     seq_cnt += 1
        #     frame_files  = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith((".jpg"))]
        #     depth_files  = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith((".exr"))]
        #     camera_files = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith((".npz"))]
        #     num_imgs = len(frame_files)
        #     # cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        #     # print(seq, len(frame_files), cut_off, not self.allow_repeat)
        #     # if num_imgs < cut_off:
        #     ids = list(np.arange(num_imgs) + offset)
        #     self.scene_img_list.append(ids)
        #     self.scenes.append(seq)
        #     self.images.extend([osp.join(seq, ff) for ff in frame_files])
        #     self.depths.extend([osp.join(seq, ff) for ff in depth_files])
        #     self.cameras.extend([osp.join(seq, ff) for ff in camera_files])
        #     self.start_img_ids.extend(ids[: num_imgs - self.num_views + 1])
        #     offset += num_imgs
        #     scene_id += 1


        list_data = []
        for dir in sorted(os.listdir(split_dir)):
            for subdir in sorted(os.listdir(f"{self.ROOT}/{dir}/")):
                list_data.append(f"{self.ROOT}/{dir}/{subdir}/image_02/data")
                list_data.append(f"{self.ROOT}/{dir}/{subdir}/image_03/data")

        seq_cnt = 0
        for seq in list_data:
            seq_cnt += 1
            split = seq.split('rgb')[1].split('/')
            depth_seq = f"{seq.split('rgb')[0]}/depth/{split[1]}/{split[2]}/proj_depth/groundtruth/{split[-2]}/"
            if os.path.exists(depth_seq):
                frame_files = [f for f in sorted(os.listdir(f"{seq}")) if f.lower().endswith((".png"))]
                depth_files = [f for f in sorted(os.listdir(f"{depth_seq}")) if f.lower().endswith((".png"))]
                
                common_files = list(set(frame_files) & set(depth_files))
                num_imgs = len(common_files)
                ids = list(np.arange(num_imgs) + offset)
                self.scene_img_list.append(ids)
                self.scenes.append(seq)
                self.images.extend([osp.join(seq, ff) for ff in common_files])
                self.depths.extend([osp.join(depth_seq, ff) for ff in common_files])
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
        # print(img_list_selected[0])
        # scene_dir = os.path.dirname(img_list_selected[0])
        scene_dir = os.path.dirname(os.path.dirname(img_list_selected[0]))

        views: List[Dict[str, Any]] = []
        for i in range(self.num_views):
            img_path = img_list_selected[i]
            depth_path = depth_list_selected[i]

            scene_name = str(Path(*Path(img_path).parts[-4:-1]))
            image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            depth = read_depth(depth_path)
            # save_depth(depth, './depthoriginal.png')

            rng = np.random.default_rng(seed=42)
            intrinsic = np.array([
                [560.0, 0.0, 256.0],
                [0.0, 560.0, 256.0],
                [0, 0, 1.0]
            ])
            extrinsic = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            image, depth = scale_invariant_center_crop(image, depth, W, H)
            # save_depth(depth, './depth.png')
            # cv2.imwrite('depthimg.png', (np.array(image).astype(np.float32)).astype(np.uint8))

            image = np.array(image).astype(np.float32) / 255.0

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

