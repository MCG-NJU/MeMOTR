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
import os
import torch
import torch.nn as nn

from .ops.modules import MSDeformAttn
from .utils import get_activation_layer, get_clones, pos_to_pos_embed
from utils.utils import inverse_sigmoid
from .mlp import MLP


class DeformableDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, merge_det_track_layer: int = 0,
                 n_det_queries: int = 300, d_model: int = 256,
                 use_checkpoint: bool = False,
                 use_dab: bool = False,
                 visualize: bool = False):
        super(DeformableDecoder, self).__init__()
        self.layers = get_clones(module=decoder_layer, n=num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.merge_det_track_layer = merge_det_track_layer
        self.n_det_queries = n_det_queries
        self.d_model = d_model
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize
        # For DAB-DETR:
        if self.use_dab:
            self.query_scale = MLP(
                input_dim=self.d_model,
                hidden_dim=self.d_model,
                output_dim=self.d_model,
                num_layers=2
            )
            self.ref_point_head = MLP(
                input_dim=self.d_model*2,
                hidden_dim=self.d_model,
                output_dim=self.d_model,
                num_layers=2
            )

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos, query_mask, src_padding_mask):
        """
        Args:
            tgt:                    query tgt, (B, Nq, C).
            reference_points:       query reference points, (B, Nq, 2/4).
            src:                    key/value, (B, Nk, C), Nk = sum(W_l * H_l).
            src_spatial_shapes:     key/value spatial shape.
            src_level_start_index:  key/value begin.
            src_valid_ratios:       kay/value legal ratios.
            query_pos:              query pos, (B, Nq, C)
            query_mask:             query mask, (B, Nq), for query padding.
            src_padding_mask:       key/value's mask from NestedTensor, (B, Nk)

        Returns: if inter is True:  (n_layers, B, Nq, C), (n_layers, B, Nq, 4)
                 else:              (B, Nq, C), (B, Nq, 4)
        """
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_queries = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.use_dab:
                anchor_embed = pos_to_pos_embed(
                    pos=reference_points_input[:, :, 0, :],
                    num_pos_feats=self.d_model//2
                )
                raw_query_pos = self.ref_point_head(anchor_embed)
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            intermediate_queries.append(output)
            if self.visualize:
                os.makedirs("./outputs/visualize_tmp/decoder/", exist_ok=True)
                torch.save(reference_points[0, :300, :].cpu(),
                           f"./outputs/visualize_tmp/decoder/detection_ref_pts_layer_{lid}.tensor")
                torch.save(reference_points[0, 300:, :].cpu(),
                           f"./outputs/visualize_tmp/decoder/track_ref_pts_layer_{lid}.tensor")

            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    query_mask,
                    src_padding_mask,
                    (lid >= self.merge_det_track_layer),
                    use_reentrant=False
                )
            else:
                output = layer(
                    tgt=output,
                    query_pos=query_pos,
                    reference_points=reference_points_input,
                    src=src,
                    src_spatial_shapes=src_spatial_shapes,
                    level_start_index=src_level_start_index,
                    query_mask=query_mask,
                    src_padding_mask=src_padding_mask,
                    merge_det_track=(lid >= self.merge_det_track_layer)
                )   # (B, Nq, C)

            if self.visualize:
                os.system(f"mv ./outputs/visualize_tmp/decoder/attn_score.tensor "
                          f"./outputs/visualize_tmp/decoder/attn_score_layer_{lid}.tensor")
                os.system(f"mv ./outputs/visualize_tmp/decoder/sampling_locations.tensor "
                          f"./outputs/visualize_tmp/decoder/sampling_locations_layer_{lid}.tensor")

            # hack implementation for iterative bounding box refinement.
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp  # (B, Nq, 4)
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                if lid < self.merge_det_track_layer:
                    reference_points = torch.cat((new_reference_points[:, :self.n_det_queries, :].detach(),
                                                  reference_points[:, self.n_det_queries:, :]), dim=1)
                else:
                    reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            # (n_layers, B, Nq, C), (n_layers, B, Nq, 4)
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), \
                   torch.stack(intermediate_queries)
        else:
            # (B, Nq, C), (B, Nq, 4)
            raise NotImplementedError(f"Not Support for no Inter Outputs.")


class DeformableDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="ReLU",
                 n_levels=4, n_heads=8, n_points=4,
                 sigmoid_attn=False, extra_track_attn=False, n_det_queries=300,
                 visualize: bool = False):
        """
        Args:
            d_model:
            d_ffn:
            dropout:
            activation:
            n_levels:
            n_heads:
            n_points:
            sigmoid_attn:
            extra_track_attn:
            n_det_queries:
            visualize:
        """
        super(DeformableDecoderLayer, self).__init__()
        self.visualize = visualize
        self.n_det_queries = n_det_queries
        self.n_heads = n_heads
        # Self Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Cross Attention
        self.cross_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            sigmoid_attn=sigmoid_attn,
            visualize=visualize
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ffn)
        self.activation = get_activation_layer(activation=activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_ffn, out_features=d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # For track query, update track query embedding.
        self.extra_track_attn = extra_track_attn    # we do not use extra track attn.
        if self.extra_track_attn:
            self.track_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_self_attn(self, tgt, query_pos, query_mask):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, tgt, key_padding_mask=query_mask)
        if self.visualize:
            torch.save(_[0, :, :].cpu(),
                       "./outputs/visualize_tmp/decoder/attn_score.tensor")
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def forward_track_attn(self, tgt, query_pos, query_mask):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > self.n_det_queries:
            tgt2, _ = self.track_attn(q[:, self.n_det_queries:], k[:, self.n_det_queries:], tgt[:, self.n_det_queries:],
                                      key_padding_mask=query_mask[:, self.n_det_queries:])
            tgt2 = self.norm4(tgt[:, self.n_det_queries:] + self.dropout5(tgt2))
            tgt = torch.cat([tgt[:, :self.n_det_queries], tgt2], dim=1)
        return tgt

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(
            self.dropout3(
                self.activation(
                    self.linear1(tgt)
                )
            )
        )
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points,
                src, src_spatial_shapes, level_start_index, query_mask, src_padding_mask=None,
                merge_det_track=False):
        """
        Args:
            tgt:                    (B, Nq, C)
            query_pos:              (B, Nq, C)
            reference_points:       (B, Nq, src_layers, 2/4)
            src:
            src_spatial_shapes:
            level_start_index:
            query_mask:             (B, Nq)
            src_padding_mask:
            merge_det_track:
        Returns:

        """
        if merge_det_track is False:
            track_tgt = tgt[:, self.n_det_queries:, :]      # (B, Nq_track, C)
            tgt = tgt[:, :self.n_det_queries, :]            # (B, Nq_det, C)
            query_pos = query_pos[:, :self.n_det_queries, :]
            reference_points = reference_points[:, :self.n_det_queries, :, :]
            query_mask = query_mask[:, :self.n_det_queries]

        if self.extra_track_attn:
            tgt = self.forward_track_attn(tgt=tgt, query_pos=query_pos, query_mask=query_mask)
        tgt = self.forward_self_attn(tgt=tgt, query_pos=query_pos, query_mask=query_mask)
        # Cross Attention
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            reference_points=reference_points,
            input_flatten=src,
            input_spatial_shapes=src_spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=src_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # FFN
        tgt = self.forward_ffn(tgt)

        if merge_det_track is False:
            tgt = torch.cat((tgt, track_tgt), dim=1)

        return tgt
