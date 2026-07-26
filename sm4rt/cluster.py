# Copyright (c) 2025 IGGT. This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
# Source: https://github.com/lifuguan/IGGT_official

import torch
from typing import Union, Tuple
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
# from hdbscan import HDBSCAN
from cuml.cluster.hdbscan import HDBSCAN

def cluster_features_to_masks_mv(
    feature: Union[torch.Tensor, np.ndarray],
    apply_colormap: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    对多视图特征图进行聚类，并选择性地生成彩色可视化掩码。

    现在支持将所有视图的特征合并后一起做DBSCAN聚类，这样同类别的物体在不同视图中会分配到相同的类别ID和颜色。

    Args:
        feature_map (Union[torch.Tensor, np.ndarray]):
            输入的特征图，形状应为 (N, H, W, C)。
        apply_colormap (bool, optional):
            如果为True，函数将额外返回一个用于可视化的彩色掩码图。默认为False。
        **kwargs:
            传递给聚类算法的参数。
            - 若 method='kmeans', 则需要提供 n_clusters (例如: n_clusters=5)。
            - 若 method='dbscan', 则需要提供 eps 和 min_samples (例如: eps=0.5, min_samples=50)。

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - 如果 apply_colormap=False (默认):
              返回一个 np.ndarray，形状为 (N, H, W)，其中的整数值代表类别ID。
            - 如果 apply_colormap=True:
              返回一个元组 (masks, colored_masks)，其中：
                - masks: (N, H, W) 的整数类别ID掩码。
                - colored_masks: (N, H, W, 3) 的彩色RGB掩码，`uint8` 类型 (0-255)。
    """
    feature_map = feature.clone()
    if not (isinstance(feature_map, (torch.Tensor, np.ndarray)) and feature_map.ndim == 4):
        raise ValueError("输入特征图必须是形状为 (N, H, W, C) 的4D Tensor或NumPy数组。")

    n, h, w, c = feature_map.shape

    # feature_map[...,3] -= 0.5

    if isinstance(feature_map, torch.Tensor):
        feature_map_np = feature_map.cpu().numpy()
    else:
        feature_map_np = feature_map



    # #####  Tiny ######
    # CLUSTERING_CONFIG = {
    #     'eps': 0.0001,
    #     'min_samples': 10,
    #     'min_cluster_size': 50,
    #     'knn_k': 20
    # }
    # #####  Small ######
    CLUSTERING_CONFIG = {
        'eps': 0.005,
        'min_samples': 50,
        'min_cluster_size': 500,
        'knn_k': 20
    }
    #####  Medium ######
    # CLUSTERING_CONFIG = {
    #     'eps': 0.01,
    #     'min_samples': 100,
    #     'min_cluster_size': 500,
    #     'knn_k': 20
    # }
    #####  Large ######
    # CLUSTERING_CONFIG = {
    #     'eps': 0.06,
    #     'min_samples': 100,
    #     'min_cluster_size': 500,
    #     'knn_k': 20
    # }

    print(f"对所有视图特征合并后做DBSCAN聚类 (N={n}, H={h}, W={w}, C={c}) ...")
    all_pixels = feature_map_np.reshape(-1, c)  # (N*H*W, C)
    hdbscan_cluster = HDBSCAN(
        cluster_selection_epsilon=CLUSTERING_CONFIG["eps"],
        min_samples=CLUSTERING_CONFIG["min_samples"],
        min_cluster_size=CLUSTERING_CONFIG["min_cluster_size"],
        allow_single_cluster=False,
    ).fit(all_pixels)
    all_labels = hdbscan_cluster.labels_  # (N*H*W,)
    invalid_label_mask = all_labels == -1

    # 利用KNN对无效标签(-1)的像素进行掩码分配
    if invalid_label_mask.sum() > 0:
        if invalid_label_mask.sum() == len(invalid_label_mask):
            # 全部为无效点，全部设为0
            all_labels = np.zeros_like(all_labels)
        else:
            valid_pixels = all_pixels[~invalid_label_mask]
            invalid_pixels = all_pixels[invalid_label_mask]
            valid_labels = all_labels[~invalid_label_mask]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(valid_pixels)
            distances, indices = nbrs.kneighbors(invalid_pixels)
            indices = indices[:, 0]
            all_labels[invalid_label_mask] = valid_labels[indices]

    # 还原为 (N, H, W)
    masks = all_labels.reshape(n, h, w)
    print("DBSCAN聚类完成。")

    if not apply_colormap:
        return masks
    else:
        print("正在应用颜色图...")
        # 全局分配颜色，保证同一label在所有视图中颜色一致
        unique_labels = np.unique(masks)
        unique_labels_no_noise = unique_labels[unique_labels != -1]
        n_colors = len(unique_labels_no_noise)
        cmap = plt.colormaps.get_cmap('jet')
        color_map = {
            label: list(cmap(j / (n_colors - 1))[:3]) if n_colors > 1 else list(cmap(0.5)[:3])
            for j, label in enumerate(unique_labels_no_noise)
        }
        color_map[-1] = [0, 0, 0]
        colored_masks = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            labels = masks[i].reshape(-1)
            colored_pixels = np.array([color_map[label] for label in labels])
            colored_pixels_uint8 = (colored_pixels * 255).astype(np.uint8)
            colored_masks[i] = colored_pixels_uint8.reshape(h, w, 3)
        return masks, colored_masks
    
