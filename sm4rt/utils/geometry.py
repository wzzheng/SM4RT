# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np


def unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:

    world_points_list = []
    cam_coords_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, cur_cam_coords_points, _ = depth_to_world_coords_points(depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx])
        world_points_list.append(cur_world_points)
        cam_coords_points_list.append(cur_cam_coords_points)
    # world_points_array = np.stack(world_points_list, axis=0)
    # cam_coords_points_array = np.stack(cam_coords_points_list, axis=0)
    world_points_array = torch.stack(world_points_list, dim=0).unsqueeze(0)
    cam_coords_points_array = torch.stack(cam_coords_points_list, dim=0).unsqueeze(0)
    return world_points_array, cam_coords_points_array


def depth_to_world_coords_points(depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-8,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth_map is None:
        return None, None, None
    point_mask = depth_map > eps

    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    world_coords_points = torch.matmul(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    return world_coords_points, cam_coords_points, point_mask


# def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     H, W = depth_map.shape
#     assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
#     assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

#     fu, fv = intrinsic[0, 0], intrinsic[1, 1]
#     cu, cv = intrinsic[0, 2], intrinsic[1, 2]
#     u, v = np.meshgrid(np.arange(W), np.arange(H))

#     x_cam = (u - cu) * depth_map / fu
#     y_cam = (v - cv) * depth_map / fv
#     z_cam = depth_map

#     cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
#     return cam_coords

def depth_to_cam_coords_points(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    u, v = torch.meshgrid(torch.arange(W, device=depth_map.device), torch.arange(H, device=depth_map.device), indexing='xy')
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1).float()
    return cam_coords



def closed_form_inverse_se3(se3, R=None, T=None):
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)
    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
    inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right
    return inverted_matrix
