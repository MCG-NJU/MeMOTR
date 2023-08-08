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
import math
import torch.nn as nn

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from typing import List

from .deformable_encoder import DeformableEncoderLayer, DeformableEncoder
from .deformable_decoder import DeformableDecoderLayer, DeformableDecoder
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 n_feature_levels=4, n_heads=8,
                 n_enc_points=4, n_dec_points=4,
                 n_enc_layers=6, n_dec_layers=6,
                 merge_det_track_layer=0,
                 dropout=0.1, activation="ReLU",
                 return_intermediate_dec=False,
                 n_det_queries=300,
                 extra_track_attn=False,
                 two_stage=False, two_stage_num_proposals=300,
                 use_checkpoint: bool = False,
                 checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False):
        """
        Args:
            d_model:
            d_ffn:
            n_feature_levels:
            n_heads:
            n_enc_points:
            n_dec_points:
            n_enc_layers:
            n_dec_layers:
            dropout:
            activation:
            return_intermediate_dec:
            n_det_queries:
            extra_track_attn:
            two_stage:
            two_stage_num_proposals:
            visualize
        """
        super(DeformableTransformer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize

        encoder_layer = DeformableEncoderLayer(
            d_model=d_model, d_ffn=d_ffn,
            dropout=dropout, activation=activation,
            n_levels=n_feature_levels, n_heads=n_heads,
            n_points=n_enc_points, sigmoid_attn=False
        )
        decoder_layer = DeformableDecoderLayer(
            d_model=d_model, d_ffn=d_ffn,
            dropout=dropout, activation=activation,
            n_levels=n_feature_levels, n_heads=n_heads,
            n_points=n_dec_points, sigmoid_attn=False,
            extra_track_attn=extra_track_attn, n_det_queries=n_det_queries,
            visualize=self.visualize
        )

        self.encoder: DeformableEncoder = DeformableEncoder(encoder_layer=encoder_layer, num_layers=n_enc_layers,
                                                            use_checkpoint=(self.use_checkpoint and
                                                                            self.checkpoint_level == 1))
        self.decoder: DeformableDecoder = DeformableDecoder(decoder_layer=decoder_layer, num_layers=n_dec_layers,
                                                            return_intermediate=return_intermediate_dec,
                                                            merge_det_track_layer=merge_det_track_layer,
                                                            n_det_queries=n_det_queries,
                                                            d_model=self.d_model,
                                                            use_checkpoint=self.use_checkpoint,
                                                            use_dab=self.use_dab,
                                                            visualize=self.visualize)

        self.level_embed = nn.Parameter(torch.Tensor(n_feature_levels, d_model))

        if two_stage:
            # 目前不关心 two stage 的情况
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if use_dab:
                pass
            else:
                self.reference_points = nn.Linear(d_model, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m.reset_parameters()
        if not self.two_stage:
            if self.use_dab:
                pass
            else:
                xavier_uniform_(self.reference_points.weight.data, gain=1.0)
                constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    @staticmethod
    def get_proposal_pos_embed(proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_valid_ratio(mask):
        """
        Args:
            mask: NestedTensor's mask

        Returns:

        """
        _, H, W = mask.shape                        # (B, H, W)
        valid_H = torch.sum(~mask[:, :, 0], 1)      # (B, )
        valid_W = torch.sum(~mask[:, 0, :], 1)      # (B, )
        valid_ratio_h = valid_H.float() / H         # (B, )
        valid_ratio_w = valid_W.float() / W         # (B, )
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio                          # (B, 2)

    def forward(self, srcs: List[torch.Tensor], masks: List[torch.Tensor],
                pos_embeds: List[torch.Tensor], query_embed, ref_pts, query_mask):
        assert self.two_stage or query_embed is not None

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            # src.shape = (B, C, H, W) in lvl level.
            # mask.shape = (B, H, W) in lvl level.
            # pos_embed.shape = (B, C, H, W) in lvl level.
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            src = src.flatten(2).transpose(1, 2)                # (B, H*W, C)
            mask = mask.flatten(1)                              # (B, H*W)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # (B, H*W, C), same as src.
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    # (B, H*W, C)
            spatial_shapes.append(spatial_shape)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)   # (n_levels, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))                          # (n_levels, )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)         # (B, n_levels, 2)

        # Encoder:
        if self.use_checkpoint and (self.checkpoint_level == 2 or self.checkpoint_level == 3):
            from torch.utils.checkpoint import checkpoint
            memory = checkpoint(self.encoder, src_flatten, spatial_shapes, level_start_index,
                                valid_ratios, lvl_pos_embed_flatten, mask_flatten, use_reentrant=False)
        else:
            memory = self.encoder(
                src=src_flatten, spatial_shapes=spatial_shapes,
                level_start_index=level_start_index, valid_ratios=valid_ratios,
                pos=lvl_pos_embed_flatten, padding_mask=mask_flatten
            )   # (B, sum(W_l * H_l), C) = (B, N, C)
        bs, _, c = memory.shape
        if self.two_stage:
            raise RuntimeError(f"Do not support two stage model for Deformable Transformer.")
        else:
            if self.use_dab:
                tgt = query_embed
                query_embed = None
            else:
                query_embed, tgt = torch.split(query_embed, c, dim=2)   # (B, Nq, C), (B, Nq, C)
            assert ref_pts is not None, "ref_pts should not be None."
            reference_points = ref_pts.sigmoid()
            init_reference_points = reference_points

        output, res_reference_points, inter_queries = self.decoder(
            tgt=tgt,
            reference_points=init_reference_points,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=query_embed,
            query_mask=query_mask,
            src_padding_mask=mask_flatten
        )
        # if inter is Ture, shape = (n_layers, B, Nq, C), (n_layers, B, Nq, 4), (n_layer, B, Nq, 2C)
        # else, shape = (B, Nq, C), (B, Nq, 4)
        return output, init_reference_points, res_reference_points, inter_queries
        # output:   if inter is Ture, (n_layers, B, Nq, C)
        #           else,             (B, Nq, C)
        # init_reference_points, (B, Nq, 2/4)
        # res_reference_points: if inter is True, (n_layers, B, Nq, 4)
        #                       else,             (B, Nq, 4)

    def get_d_model(self):
        return self.d_model

    def get_n_dec_layers(self):
        return self.decoder.num_layers

    def set_refine_bbox_embed(self, bbox_embed: nn.Module):
        self.decoder.bbox_embed = bbox_embed
        return


def build(config: dict):
    return DeformableTransformer(
        d_model=config["HIDDEN_DIM"],
        d_ffn=config["FFN_DIM"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        n_heads=config["NUM_HEADS"],
        n_enc_points=config["NUM_ENC_POINTS"],
        n_dec_points=config["NUM_DEC_POINTS"],
        n_enc_layers=config["NUM_ENC_LAYERS"],
        n_dec_layers=config["NUM_DEC_LAYERS"],
        merge_det_track_layer=0 if "MERGE_DET_TRACK_LAYER" not in config else config["MERGE_DET_TRACK_LAYER"],
        dropout=config["DROPOUT"],
        activation=config["ACTIVATION"],
        return_intermediate_dec=config["RETURN_INTER_DEC"],
        n_det_queries=config["NUM_DET_QUERIES"],
        extra_track_attn=config["EXTRA_TRACK_ATTN"],
        two_stage=False,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"]
    )

