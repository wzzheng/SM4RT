import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sm4rt.utils.transform import pose_encoding_to_extri_intri, affine_inverse, as_homogeneous
from sm4rt.dinov2 import DINOV2
from sm4rt.heads.dualdpt import DualDPT
from sm4rt.heads.dpt_head import DPTHead

from sm4rt.utils.geometry import unproject_depth_map_to_point_map

import logging
logger = logging.getLogger(__name__)
from sm4rt.layers.block import Block
from sm4rt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from sklearn.decomposition import PCA

from sparsemax import Sparsemax
from sm4rt.se3_ops import se3_exp_map
from sm4rt.cluster import cluster_features_to_masks_mv


class CameraDec(nn.Module):
    def __init__(self, dim_in=1536):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU(),)
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat, camera_encoding=None, *args, **kwargs):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        out_t = self.fc_t(feat.float()).reshape(B, N, 3)
        if camera_encoding is None:
            out_qvec = self.fc_qvec(feat.float()).reshape(B, N, 4)
            out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        else:
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., -2:]
        pose_enc = torch.cat([out_t, out_qvec, out_fov], dim=-1)
        return pose_enc


class TwistDec(nn.Module):
    def __init__(self, dim_in=1536):
        super().__init__()
        self.d_model = dim_in

        self.regression_head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.regression_head_v = nn.Sequential(nn.Linear(256, 3))
        self.regression_head_w = nn.Sequential(nn.Linear(256, 3))

    def forward(self, feat):
        motion_token = feat
        all_twists = []
        for t in range(motion_token.shape[1]):  # t = 0, 1, ..., T-1
            # all_outputs.append(output)
            output = motion_token[:,t]
            twists_tmp = self.regression_head(output)  # [B, K, 6]
            twists_v = self.regression_head_v(twists_tmp)  # [B, K, 6]
            twists_w = self.regression_head_w(twists_tmp)  # [B, K, 6]
            twists = torch.cat([twists_v, twists_w], dim=-1)
            all_twists.append(twists)

        twists_seq = torch.stack(all_twists, dim=1)  # [B, T, K, 6]
        return twists_seq
    

class MotionBaseDecoder(nn.Module):
    def __init__(
        self,
        patch_size=14, 
        embed_dim=1536, 
        depth=4, 
        num_heads=16, 
        mlp_ratio=4.0,
        num_motion_tokens=10, 
        block_fn=Block,
        qkv_bias=True, proj_bias=True, ffn_bias=True, qk_norm=True, 
        rope_freq=100, 
        init_values=0.01,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_motion_tokens = num_motion_tokens

        self.cross_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=qkv_bias,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cross_attn_ffn_norm = nn.LayerNorm(embed_dim)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
            )
            for _ in range(depth)
        ])
        
        self.motion_token = nn.Parameter(torch.randn(1, 10, embed_dim))
        self.reserved_token = nn.Parameter(torch.randn(1, 5, embed_dim))
        self.scale = nn.Parameter(torch.randn(1) * 0.5)

        self.feature_adapters = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(4)
        ])

    def get_temporal_encoding(self, S, device):
        dim = self.embed_dim
        position = torch.arange(S, device=device).unsqueeze(1) # [S, 1]
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-np.log(10000.0) / dim))
        pe = torch.zeros(S, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)

    def forward(self, tokens: torch.Tensor, images: torch.Tensor, patch_start_idx: int, track_query_idx = 0,) -> torch.Tensor:
        tokens = [tok[:,:,2:,] for tok in tokens]
        _, _, _, H, W = images.shape
        token_0 = tokens[0]
        device = token_0.device
        B, S, P, C = token_0.shape

        motion_tokens = torch.cat([
            self.motion_token.repeat(S, 1, 1),
            self.reserved_token.repeat(S, 1, 1), # Not for Motion Decoder
        ], dim=1) + self.scale * self.get_temporal_encoding(S, device) # [S, 1, D]
        
        feats_flat_0 = token_0.view(S, P, C)
        feats_list = [t.view(S, P, C) for t in tokens]
        attn_out, _ = self.cross_attn_layer(
            query=motion_tokens, 
            key=feats_list[0], 
            value=feats_list[0], 
            need_weights=False
        )
        motion_tokens = self.cross_attn_norm(motion_tokens + attn_out)
        ffn_out = self.cross_attn_ffn(motion_tokens)
        motion_tokens = self.cross_attn_ffn_norm(motion_tokens + ffn_out)

        tokens = torch.cat([motion_tokens, feats_flat_0], dim=1)

        pos = None
        if self.rope is not None:
            pos_patch = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)
            pos_motion = torch.zeros(B * S, 10, 2, device=device, dtype=pos_patch.dtype)
            pos = torch.cat([pos_motion, pos_patch], dim=1)

        output_list = []
        for i, block in enumerate(self.blocks):
            if i > 0 and i < len(feats_list):
                m_part = tokens[:, :15, :]
                p_part = tokens[:, 15:, :]
                p_part = p_part + self.feature_adapters[i](feats_list[i])
                tokens = torch.cat([m_part, p_part], dim=1)

            tokens = block(tokens, pos=pos)
            output_list.append(tokens.unsqueeze(0))
            
        return output_list


class SM4RT(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = DINOV2(alt_start=13, qknorm_start=13, rope_start=13,)
        self.head = DualDPT(dim_in=3072, output_dim=2, features=256,)
        self.cam_dec = CameraDec(dim_in=3072)

        self.motion_base_decoder = MotionBaseDecoder()
        self.assign_head = DPTHead(
            dim_in=1536, output_dim=11, activation="linear",
            conf_activation="expp1", intermediate_layer_idx=[0, 1, 2, 3],
        )
        self.twist_decoder = TwistDec(dim_in=1536)
        self.sparsemax = Sparsemax(dim=-1)


    def forward(self, views, adaptive_grouping=False, filter_large_diff=False, aux=None, **kwargs):
        video_name = None
        if 'video_name' in views[0].keys():
            video_name = views[0]['video_name'][0]
            
        img = torch.stack([view["img"] for view in views], dim=1)
        x = img
        
        feats, _ = self.backbone(x, ref_view_strategy='first',)
        patch_token = torch.cat([feats[i][0] for i in range(4)], dim=0).unsqueeze(0)
        cam_token = torch.cat([feats[i][1] for i in range(4)], dim=0).unsqueeze(0)
        time_token = torch.cat([feats[i][2] for i in range(4)], dim=0).unsqueeze(0)
        S, H, W = x.shape[1], x.shape[-2], x.shape[-1]

        with torch.autocast(device_type=next(self.parameters()).device.type, dtype=torch.float32):
            predictions = self.head(patch_token, H, W, patch_start_idx=0)
            pose_enc = self.cam_dec(cam_token[:,-1])
            predictions["pose_enc"] = pose_enc
        
        feature_list = [torch.cat([cam_token[:,i].unsqueeze(2), time_token[:,i].unsqueeze(2), patch_token[:,i]], dim=2,)[..., :1536] for i in range(4)]
        aggregated_track_tokens_list = self.motion_base_decoder(feature_list, images=x, patch_start_idx=2, track_query_idx=0)
        
        with torch.autocast(device_type=next(self.parameters()).device.type, dtype=torch.float32):
            assign, _ = self.assign_head(aggregated_track_tokens_list, images=x, patch_start_idx=15, frames_chunk_size=4)
            assign_0 = self.sparsemax(assign[:,0])
            assign_original_0 = assign_0.clone()

            feat = assign_0[0] # shape: [H, W, 10]
            h, w, c = feat.shape
            feat_flat = feat.reshape(-1, c).detach().cpu().numpy()  # shape: (518*518, 10)
            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feat_flat)
            feat_pca = feat_pca.reshape(h, w, 3)
            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min())
            img_to_save = (feat_pca * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_to_save)
            img_pil.save('./output/assign.png')

            if adaptive_grouping:
                class_mask = cluster_features_to_masks_mv(assign_0)[0]
                mask_image = (class_mask).astype(np.uint8)
                mask_image_tensor = torch.tensor(mask_image)
                unique_vals, counts = torch.unique(mask_image_tensor.contiguous().long(), return_counts=True)
                pil_image = Image.fromarray((mask_image.astype(np.float32) / (len(unique_vals) - 1) * 255).astype(np.uint8), mode='L')
                pil_image.save('./output/classmask.png')
                for val, count in zip(unique_vals, counts):
                    bool_mask = (mask_image_tensor == val)
                    # save_image((~bool_mask).float(), f"mask_{val}.png")
                    class_mean = assign_0[:, bool_mask, :].mean(dim=1)
                    assign_0[:, bool_mask, :] = class_mean
                if filter_large_diff:
                    diff = torch.abs((assign_0 - assign_original_0)).sum(dim=-1)[0]
                    diff_invalid = (diff > 0.05)
                    assign_0[:,diff_invalid] = assign_original_0[:, diff_invalid]

            twist = self.twist_decoder(aggregated_track_tokens_list[-1][:,:,:10])
            if not self.training:
                twist = twist - twist[:,:1]
            twist_weighted = torch.einsum('bhwn,bsnc->bshwc', assign_0, twist)
            twist_weighted_mat = se3_exp_map(twist_weighted.reshape(S*H*W, 6)).reshape(1, S, H, W, 4, 4)

        B, N, H, W = predictions['depth'].shape
        c2w, ixt = pose_encoding_to_extri_intri(predictions.pose_enc, (H, W))
        predictions.extrinsics_token = affine_inverse(c2w)
        predictions.intrinsics_token = ixt
        depth_tensor = predictions["depth"]
        extrinsic_tensor = predictions["extrinsics_token"]
        intrinsic_tensor = predictions["intrinsics_token"]

        wp, _ = unproject_depth_map_to_point_map(depth_tensor[0][..., None], extrinsic_tensor[0],  intrinsic_tensor[0])

        if self.training:
            wp0_hom = torch.cat([views[0]['world_pt_518'][0], torch.ones(H, W, 1).to(wp.device)], dim=-1)  # [518,518,4]
        else:
            wp0_hom = torch.cat([wp[0,0], torch.ones(H, W, 1).to(wp.device)], dim=-1)  # [518,518,4]

        wp_track = torch.matmul(twist_weighted_mat, wp0_hom.unsqueeze(-1)).squeeze(-1)[...,:3]
        output_list = []
        for i in range(N):
            extrinsic_w2c = extrinsic_tensor[0][i]
            extrinsic_c2w = affine_inverse(as_homogeneous(extrinsic_w2c))
            intrinsic_matrix = intrinsic_tensor[0][i]
            output_list.append({
                "twist_se3": twist_weighted_mat[:,i], 
                "wp": wp[0,i], 
                "wp_track": wp_track[0,i],
                'assign': assign_0[0],
                "extrinsic": extrinsic_c2w,
                "intrinsic": intrinsic_matrix,
                'twist': twist_weighted[:,i],
            })
        return output_list
        
    