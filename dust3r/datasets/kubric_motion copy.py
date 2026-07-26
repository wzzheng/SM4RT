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
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

import logging
# logging.getLogger('PIL').setLevel(logging.WARNING)
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
        # img_r = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
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


LIST = [4716, 2952, 1383, 2677, 1060, 5142, 5179, 1581, 2592, 777, 5089, 5550, 4215, 489, 2484, 1544, 1915, 3842, 2036, 43, 1240, 1957, 2782, 1607, 5525, 4973, 4682, 1317, 3417, 4719, 3459, 2799, 673, 2458, 585, 4932, 4191, 305, 4360, 1263, 5412, 4964, 1859, 63, 5365, 2945, 2130, 2344, 1660, 450, 4854, 118, 697, 2842, 5120, 287, 1784, 1574, 2159, 1623, 1146, 1400, 1064, 2100, 109, 628, 2925, 431, 1269, 4726, 4970, 3045, 372, 3808, 5244, 3410, 1451, 1251, 3620, 5091, 4428, 5218, 3907, 1039, 721, 2605, 2115, 5282, 4217, 4583, 1962, 3957, 1617, 5153, 1068, 4694, 3490, 3996, 5428, 191, 782, 4183, 5192, 3448, 5353, 1729, 1156, 5386, 527, 703, 4044, 1367, 203, 2621, 4662, 951, 5181, 4661, 893, 2852, 2519, 276, 4721, 204, 3719, 1336, 5307, 2417, 4744, 1498, 4189, 709, 1775, 1342, 3333, 787, 2534, 4323, 386, 1385, 2875, 3973, 1884, 1939, 189, 2751, 4819, 155, 3584, 1319, 4131, 153, 1862, 4734, 2032, 888, 2826, 1591, 846, 4683, 423, 1772, 3574, 3405, 2294, 103, 319, 608, 2022, 1230, 589, 1613, 3552, 2337, 4062, 3233, 159, 795, 2633, 404, 3667, 200, 349, 5146, 2407, 2203, 4985, 953, 5389, 2358, 2715, 4170, 1722, 4453, 4293, 5065, 838, 820, 4668, 1610, 2495, 577, 2692, 4332, 3777, 1458, 3251, 1880, 1937, 4059, 12, 353, 3118, 4052, 898, 4322, 3062, 2008, 1553, 4652, 671, 1625, 4687, 2429, 2517, 5018, 3613, 2576, 2540, 1463, 2067, 3878, 1228, 5617, 1460, 5253, 1037, 4258, 3186, 1006, 1827, 2802, 2237, 2307, 1046, 480, 1537, 3076, 530, 1733, 4025, 3293, 5344, 4315, 31, 2013, 4569, 2590, 1338, 134, 4937, 1782, 4136, 3778, 857, 1281, 154, 1500, 5623, 2421, 2447, 4414, 788, 1816, 3571, 2671, 1161, 460, 3962, 5495, 279, 2837, 122, 2355, 932, 2143, 2800, 4760, 456, 2161, 1906, 4730, 3843, 1330, 2259, 5076, 2696, 3037, 4960, 3420, 829, 4852, 835, 3190, 2787, 4109, 4231, 223, 430, 1504, 651, 2196, 3060, 3649, 3977, 3599, 1970, 5286, 3327, 4222, 736, 3369, 3160, 2424, 1309, 539, 3820, 3138, 3350, 935, 1010, 5393, 2405, 3136, 101, 1977, 3473, 3133, 4753, 4113, 728, 90, 1219, 2078, 2493, 5116, 3701, 4971, 4086, 2467, 2010, 3903, 4001, 5131, 4772, 1216, 2511, 1693, 2606, 2183, 2216, 961, 78, 5020, 4158, 3247, 4624, 3111, 3979, 5263, 659, 5351, 1632, 184, 4579, 1554, 2995, 2739, 3797, 2241, 1719, 977, 2463, 2165, 1425, 3972, 414, 1769, 1158, 998, 2129, 2190, 5005, 4244, 3643, 529, 2823, 5371, 2209, 4015, 856, 674, 3242, 1686, 3346, 1431, 4142, 2821, 1811, 2289, 4185, 358, 1622, 40, 613, 1072, 4172, 808, 2500, 2838, 5605, 891, 4722, 39, 1968, 2304, 257, 2169, 5658, 5519, 3157, 1524, 1510, 86, 4133, 942, 5341, 5010, 2698, 5015, 5248, 5506, 5274, 5650, 5186, 369, 4507, 1066, 45, 5200, 2999, 3968, 4986, 2777, 2017, 4817, 3375, 425, 4107, 1672, 4399, 2246, 516, 2486, 4095, 2733, 5454, 5468, 5616, 4123, 3855, 3679, 2224, 3068, 4805, 66, 2571, 1479, 4976, 2781, 3055, 4582, 410, 2849, 4853, 5188, 1280, 2111, 3493, 772, 4049, 4915, 676, 3502, 5411, 1668, 4379, 4593, 3578, 3367, 4795, 3929, 2905, 5012, 2402, 855, 2547, 665, 5100, 180, 1628, 114, 3801, 5618, 4909, 3467, 3570, 4411, 237, 1866, 160, 2825, 5572, 5522, 4880, 5402, 1518, 2074, 74, 2026, 1124, 3005, 3283, 879, 4068, 5199, 4680, 4353, 161, 2735, 1312, 3828, 5659, 5090, 451, 2213, 952, 5363, 1835, 2354, 3756, 3931, 5646, 612, 4187, 2432, 1647, 5236, 3883, 5280, 5608, 4012, 2095, 4182, 348, 2624, 1838, 3577, 3824, 3388, 3588, 1320, 20, 3137, 3685, 2226, 4650, 283, 4720, 2002, 538, 2072, 1706, 883, 2668, 4868, 4327, 2938, 4457, 3086, 3124, 1764, 4807, 2866, 3621, 1662, 2857, 1631, 2222, 1386, 4470, 768, 5384, 4148, 4202, 591, 1919, 2858, 2440, 2264, 4519, 3217, 1106, 3245, 3535, 4762, 1887, 1207, 802, 4727, 1875, 4618, 2314, 4777, 984, 5201, 2824, 1934, 3117, 370, 4393, 1139, 1259, 3180, 136, 2894, 2635, 5004, 760, 1087, 3043, 124, 1160, 3868, 4491, 3776, 3182, 2854, 195, 406, 2729, 179, 2114, 2131, 4928, 4168, 4017, 1209, 2503, 1377, 2442, 4271, 1933, 3149, 2498, 294, 2884, 228, 290, 4162, 5288, 3294, 277, 5297, 5271, 3823, 3669, 2702, 4710, 3949, 1942, 5135, 4407, 5564, 5308, 3183, 1898, 4503, 4979, 2364, 4572, 5452, 5566, 2042, 4829, 2261, 4827, 3446, 3241, 5593, 5471, 91, 1548, 2107, 3918, 1783, 4508, 3010, 1004, 4067, 762, 1266, 446, 4489, 2394, 875, 398, 1105, 3528, 227, 1418, 3169, 2195, 5292, 1172, 4789, 4641, 2773, 4468, 1824, 71, 3450, 3248, 4249, 110, 1612, 4690, 3951, 1577, 1394, 4809, 4822, 2864, 1485, 1833, 799, 2033, 4348, 2786, 4901, 900, 4016, 1499, 4748, 1398, 2425, 2926, 4670, 1981, 267, 5048, 4776, 5463, 1268, 3019, 2156, 4164, 4621, 4010, 417, 2779, 2840, 5558, 1950, 2313, 4837, 2974, 272, 4550, 3553, 937, 4208, 4584, 3930, 1640, 5635, 421, 2554, 692, 5123, 2092, 5327, 3164, 2690, 3810, 666, 3935, 2844, 3018, 4775, 637, 5598, 384, 1448, 269, 3548, 5323, 3109, 5111, 4281, 2083, 3698, 3261, 3172, 4223, 3706, 2328, 3474, 4784, 4742, 1922, 4308, 4083, 4110, 2359, 826, 5060, 1586, 1409, 2617, 5502, 3162, 5585, 2309, 3332, 4262, 2455, 3950, 1278, 4640, 578, 767, 1539, 1486, 1713, 1578, 2878, 4206, 1532, 167, 21, 4972, 779, 1417, 1676, 2944, 5028, 3065, 4541, 3752, 2718, 3993, 3303, 6, 5379, 3576, 87, 5382, 3236, 1995, 4780, 3580, 4948, 4290, 426, 24, 5545, 654, 3712, 4166, 4605, 3549, 4872, 4513, 633, 1089, 3637, 4790, 1402, 706, 750, 559, 1017, 1883, 2406, 1412, 1819, 2708, 2812, 5347, 3154, 5171, 402, 1975, 2689, 5221, 3438, 3119, 1525, 1735, 2846, 2876, 4460, 4235, 5666, 4620, 1220, 5113, 3638, 5140, 1294, 1157, 943, 1354, 1606, 1279, 5349, 1150, 4036, 1960, 1944, 4927, 2332, 3990, 1869, 3952, 5481, 3978, 174, 1464, 3341, 444, 5206, 3427, 3518, 3939, 3046, 4626, 1323, 5133, 632, 796, 4344, 1436, 4598, 2992, 65, 5581, 947, 1380, 1832, 4614, 4272, 1965, 4165, 2710, 246, 4476, 5121, 1522, 4402, 4750, 4787, 3668, 5511, 4528, 1177, 1776, 997, 4826, 2843, 2504, 2544, 2664, 2416, 1348, 185, 477, 3330, 939, 565, 3865, 645, 5484, 1765, 2285, 4034, 3271, 2211, 4325][:500]

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

        # seq_list = sorted(os.listdir(self.ROOT))
        # print(seq_list)
        seq_list = [f"{i:04d}" for i in LIST]
        # seq_list = sorted(os.listdir(self.ROOT))[::2]
        if '0073' in seq_list:
            seq_list.remove('0073')
        seq_cnt = 0
        for seq in seq_list:
            if seq_cnt % 100 == 0:
                print(seq)
            seq_cnt += 1
            if seq_cnt > self.max_sequences:
                break
            frame_path    = osp.join(self.ROOT, seq, "frames")
            seg_path      = osp.join(self.ROOT, seq, "segmentations", '000.png')
            world_pt_path = osp.join(self.ROOT, seq, f"{seq}_world_pts.pt")
            frames = [osp.join(frame_path, f"{i:03d}.png") for i in range(self.num_views)]
            # twist_path = osp.join(self.ROOT, seq, f"{seq}_twist.pt")
            scale_path = osp.join(self.ROOT, seq, f"{seq}_scale.txt")
            dense_path = osp.join(self.ROOT, seq, f"{seq}_dense.npy")

            self.store[seq] = {
                "frames": frames, 
                "seg_path": seg_path, 
                "world_pt_path": world_pt_path,
                # "twist_path": twist_path,
                "scale_path": scale_path,
                "dense_path": dense_path,
            }

        self.sequence_list = list(self.store.keys())
        self.cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        self.all_ref = []
        for seq in self.sequence_list:
            T = len(self.store[seq]["frames"])
            if T >= self.cut_off:
                self.all_ref.extend([(seq, 0)])
                # self.all_ref.extend([(seq, s) for s in range(T - self.cut_off + 1)])

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

        # W, H = self._resolutions[ar_idx]
        # print(index)

        try:
            seed = torch.randint(0, 2**32, (1,)).item()
        except Exception:
            seed = int(time.time() * 1e6) & 0xFFFFFFFF
        rng = np.random.default_rng(seed=seed)
        self._rng = rng
        seq_name, start_id = self.all_ref[index]
        frames = self.store[seq_name]["frames"]
        pos = list(range(len(frames)))[:self.num_views]

        # visibility = torch.tensor(np.load(self.store[seq_name]["dense_path"], allow_pickle=True).item()['visibility']).transpose(1,0)[:self.num_views].reshape(self.num_views, 512, 512)
        # visibility_518 = torch.nn.functional.interpolate(visibility.unsqueeze(1).float(), size=(518, 518), mode='nearest-exact').permute(0, 2, 3, 1).bool().squeeze(-1)
        # visibility_518 = ~visibility_518
        # /data1/datasets/kubric_world/kubric_world/0000/0000_dense.npy

        world_pt = torch.load(self.store[seq_name]["world_pt_path"], map_location='cpu', weights_only=False)[0,pos]
        world_pt_518 = torch.nn.functional.interpolate(world_pt.permute(0, 3, 1, 2), size=(518, 518), mode='nearest-exact').permute(0, 2, 3, 1).float()

        scale = float(open(self.store[seq_name]["scale_path"]).read().strip())
        Frame, H, W, _ = world_pt_518.shape

        track3d_disp_diff = world_pt_518 - world_pt_518[:1]
        diff = track3d_disp_diff[1:] - track3d_disp_diff[:-1]
        frame_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        norm = frame_norms.sum(dim=0)

        world_pt_518 /= scale
        track3d_disp = world_pt_518 - world_pt_518[:1]

        seg_path = self.store[seq_name]["seg_path"]
        img = Image.open(seg_path).convert('L')
        img_np = np.array(img)

        # print(img_np.dtype == np.float32 or img_np.dtype == np.float64)
        # if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        #     if img_np.max() <= 1.0:
        #         img_np = (img_np * 255).astype(np.uint8)
        #     else:
        #         img_np = img_np.astype(np.int32)
        
        seg_mask = torch.from_numpy(img_np).long()
        seg_mask_518 = torch.nn.functional.interpolate(seg_mask.unsqueeze(0).unsqueeze(0).float(), size=(518, 518), mode='nearest-exact', align_corners=None)[0,0].long()
        background = (seg_mask_518 == 0) | (norm < 0.05)
        # twist = torch.load(self.store[seq_name]["twist_path"], map_location='cpu', weights_only=False)
        # print(background.shape, norm.shape)
        # exit()

        views: List[Dict[str, Any]] = []

        for s_idx, p in enumerate(pos):
            img_path = frames[p]
            img = PIL.ImageOps.exif_transpose(PIL.Image.open(img_path)).convert("RGB")
            img = img.resize((518, 518), PIL.Image.LANCZOS)

            # print(torch.tensor(np.array(img)).max(), torch.tensor(np.array(img)).min())
            # print(ImgNorm(img).max(), ImgNorm(img).min())
            # exit()
            views.append(dict(
                img=ImgNorm(img),
                dataset=self.dataset_label,
                label=seq_name,
                instance=osp.basename(img_path),
                track3d_disp=track3d_disp[p],
                world_pt_518=world_pt_518[p],
                # vis=visibility_518[p],
                # twist=twist[p],
                background=background,
                is_metric=False,
            ))
        return views

