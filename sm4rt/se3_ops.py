# Copyright (c) 2023 Vivek Gopalakrishnan. This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
# Source: https://github.com/eigenvivek/pytorchse3

import torch


def se3_exp_map(log_T_vee, n=10):
    log_R_vee = log_T_vee[..., 3:]
    log_t_vee = log_T_vee[..., :3]

    theta = log_R_vee.norm(dim=-1, keepdim=True).unsqueeze(-1)
    
    batch_size = len(log_R_vee)
    log_R = torch.zeros(
            batch_size, 3, 3,
            dtype=log_R_vee.dtype,
            device=log_R_vee.device,
    )
    log_R[..., 0, 1] = -log_R_vee[..., 2]
    log_R[..., 0, 2] = +log_R_vee[..., 1]
    log_R[..., 1, 2] = -log_R_vee[..., 0]
    log_R[..., 1, 0] = +log_R_vee[..., 2]
    log_R[..., 2, 0] = -log_R_vee[..., 1]
    log_R[..., 2, 1] = +log_R_vee[..., 0]

    log_R_2 = torch.linalg.matrix_power(log_R, 2)


    A = torch.zeros_like(theta)
    denom_a = 1.0
    B = torch.zeros_like(theta)
    denom_b = 1.0
    C = torch.zeros_like(theta)
    denom_c = 1.0

    for i in range(11):
        if i > 0:
            denom_a *= (2 * i) * (2 * i + 1)
        denom_b *= (2 * i + 1) * (2 * i + 2)
        denom_c *= (2 * i + 2) * (2 * i + 3)
        A = A + (-1) ** i * theta ** (2 * i) / denom_a
        B = B + (-1) ** i * theta ** (2 * i) / denom_b
        C = C + (-1) ** i * theta ** (2 * i) / denom_c

    R = torch.eye(3, dtype=A.dtype, device=A.device) + A * log_R + B * log_R_2
    V = torch.eye(3, dtype=A.dtype, device=A.device) + B * log_R + C * log_R_2
    t = torch.einsum("bij, bj -> bi", V, log_t_vee)

    T = torch.zeros((len(theta), 4, 4), dtype=A.dtype, device=A.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T

