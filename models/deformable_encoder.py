# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .ops.modules import MSDeformAttn
from .utils import get_activation_layer, get_clones


class DeformableEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint: bool):
        super(DeformableEncoder, self).__init__()
        self.layers = get_clones(module=encoder_layer, n=num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            if self.use_checkpoint:
                assert len(self.layers) % 2 == 0, f"Encoder Layers must be 3x"

                def fn(x, i):
                    x = self.layers[i](x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                    x = self.layers[i + 1](x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                    x = self.layers[i + 2](x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                    return x
                if _ % 3 == 0:
                    output = checkpoint(fn, output, _, use_reentrant=False)
                else:
                    pass
            else:
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableEncoderLayer(nn.Module):
    """
    input:
    output: (B, Nq, C)
    """
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="ReLU",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super(DeformableEncoderLayer, self).__init__()

        # Self Attention
        self.self_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            sigmoid_attn=sigmoid_attn
        )   # output shape: (B, Nq, C)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ffn)
        self.activation = get_activation_layer(activation=activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_ffn, out_features=d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(
            self.dropout2(
                self.activation(
                    self.linear1(src)
                )
            )
        )
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        Args:
            src:                    (B, Nq, C)
            pos:                    (B, Nq, C)
            reference_points:       (B, Nq, n_levels, 2) for point, (B, Nq, n_levels, 4) for box
            spatial_shapes:         (n_levels, 2), as H,W
            level_start_index:      (n_levels, )
            padding_mask:           (B, \\sum_{l=0}^{L-1} H_l \\cdot W_l),
                                    True for padding elements, False for non-padding elements

        Returns:

        """
        # Self Attention
        src2 = self.self_attn(self.with_pos_embed(src, pos),
                              reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src
