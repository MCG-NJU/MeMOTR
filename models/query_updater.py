# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import math
import torch
import torch.nn as nn

from typing import List
from .utils import pos_to_pos_embed, logits_to_scores
from torch.utils.checkpoint import checkpoint

from .ffn import FFN
from .mlp import MLP
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union


class QueryUpdater(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int,
                 tp_drop_ratio: float, fp_insert_ratio: float,
                 dropout: float,
                 use_checkpoint: bool, use_dab: bool,
                 update_threshold: float, long_memory_lambda: float,
                 visualize: bool = False):
        super(QueryUpdater, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize

        self.update_threshold = update_threshold
        self.long_memory_lambda = long_memory_lambda

        self.confidence_weight_net = nn.Sequential(
            MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, num_layers=2),
            nn.Sigmoid()
        )
        self.short_memory_fusion = MLP(input_dim=2*self.hidden_dim, hidden_dim=2*self.hidden_dim,
                                       output_dim=self.hidden_dim, num_layers=2)
        self.memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_pos_head = MLP(
            input_dim=self.hidden_dim*2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                no_augment: bool = False):
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self.update_tracks_embedding(tracks=tracks)

        return tracks

    def update_tracks_embedding(self, tracks: List[TrackInstances]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold
            if self.visualize:
                os.makedirs("./outputs/visualize_tmp/query_updater/", exist_ok=True)
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/current_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, :].cpu(),
                #            "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/current_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/current_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/current_scores.tensor")
            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes[:, :2].detach().clone())

            if self.use_dab:
                query_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
                output_embed = tracks[b].output_embed
                last_output_embed = tracks[b].last_output
                long_memory = tracks[b].long_memory.detach()

                # Confidence Weight
                confidence_weight = self.confidence_weight_net(output_embed)

                # Adaptive Aggregation
                short_memory = self.short_memory_fusion(
                    torch.cat((
                        confidence_weight * output_embed,
                        last_output_embed
                    ), dim=-1)
                )

                # Query Feature Generate
                query_pos = self.query_pos_head(query_pos)
                q = short_memory + query_pos
                k = long_memory + query_pos
                tgt = output_embed
                # Attention
                tgt2 = self.memory_attn(q[None, :], k[None, :], tgt[None, :])[0][0, :]
                tgt = tgt + self.memory_dropout(tgt2)
                tgt = self.memory_norm(tgt)
                tgt = self.memory_ffn(tgt)
                # Long Memory ResNet
                query_feat = long_memory + self.query_feat_dropout(tgt)
                query_feat = self.query_feat_norm(query_feat)
                query_feat = self.query_feat_ffn(query_feat)

                # Update Long Memory
                long_memory = (1 - self.long_memory_lambda) * long_memory + \
                              self.long_memory_lambda * tracks[b].output_embed
                tracks[b].long_memory = tracks[b].long_memory * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                        long_memory * is_pos.reshape((is_pos.shape[0], 1))
                # Update Last Outputs Embedding
                tracks[b].last_output = tracks[b].last_output * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                        output_embed * is_pos.reshape((is_pos.shape[0], 1))

            else:
                raise ValueError(f"Do not support use_dab=False yet.")

            tracks[b].query_embed[is_pos] = query_feat[is_pos]

            if self.visualize:
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/next_ref_pts.tensor")
                # torch.save(tracks[b].query_embed[:, :self.hidden_dim].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_pos.tensor")
                # torch.save(tracks[b].query_embed[:, self.hidden_dim:].cpu(),
                #            "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/next_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/next_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/next_scores.tensor")

        return tracks

    def select_active_tracks(self, previous_tracks: List[TrackInstances],
                             new_tracks: List[TrackInstances],
                             unmatched_dets: List[TrackInstances],
                             no_augment: bool = False):
        tracks = []
        if self.training:
            for b in range(len(new_tracks)):
                # Update fields
                new_tracks[b].last_output = new_tracks[b].output_embed
                new_tracks[b].long_memory = new_tracks[b].query_embed
                unmatched_dets[b].last_output = unmatched_dets[b].output_embed
                unmatched_dets[b].long_memory = unmatched_dets[b].query_embed
                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = TrackInstances.cat_tracked_instances(active_tracks, unmatched_dets[b])
                    scores = torch.max(logits_to_scores(logits=active_tracks.logits), dim=1).values
                    keep_idxes = (scores > self.update_threshold) | (active_tracks.ids >= 0)
                    active_tracks = active_tracks[keep_idxes]
                    active_tracks.ids[active_tracks.iou < 0.5] = -1
                else:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = active_tracks[(active_tracks.iou > 0.5) & (active_tracks.ids >= 0)]
                    if self.tp_drop_ratio > 0.0 and not no_augment:
                        if len(active_tracks) > 0:
                            tp_keep_idx = torch.rand((len(active_tracks), )) > self.tp_drop_ratio
                            active_tracks = active_tracks[tp_keep_idx]
                    if self.fp_insert_ratio > 0.0 and not no_augment:
                        selected_active_tracks = active_tracks[
                            torch.bernoulli(
                                torch.ones((len(active_tracks), )) * self.fp_insert_ratio
                            ).bool()
                        ]
                        if len(unmatched_dets[b]) > 0 and len(selected_active_tracks) > 0:
                            fp_num = len(selected_active_tracks)
                            if fp_num >= len(unmatched_dets[b]):
                                insert_fp = unmatched_dets[b]
                            else:
                                selected_active_boxes = box_cxcywh_to_xyxy(selected_active_tracks.boxes)
                                unmatched_boxes = box_cxcywh_to_xyxy(unmatched_dets[b].boxes)
                                iou, _ = box_iou_union(unmatched_boxes, selected_active_boxes)
                                fp_idx = torch.max(iou, dim=0).indices
                                fp_idx = torch.unique(fp_idx)
                                insert_fp = unmatched_dets[b][fp_idx]
                            active_tracks = TrackInstances.cat_tracked_instances(active_tracks, insert_fp)

                if len(active_tracks) == 0:
                    device = next(self.query_feat_ffn.parameters()).device
                    fake_tracks = TrackInstances(frame_height=1.0, frame_width=1.0, hidden_dim=self.hidden_dim).to(
                        device=device)
                    if self.use_dab:
                        fake_tracks.query_embed = torch.randn((1, self.hidden_dim), dtype=torch.float,
                                                              device=device)
                    else:
                        fake_tracks.query_embed = torch.randn((1, 2 * self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.output_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    if self.use_dab:
                        fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    else:
                        fake_tracks.ref_pts = torch.randn((1, 2), dtype=torch.float, device=device)
                    fake_tracks.ids = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.matched_idx = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.boxes = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.logits = torch.randn((1, active_tracks.logits.shape[1]), dtype=torch.float, device=device)
                    fake_tracks.iou = torch.zeros((1,), dtype=torch.float, device=device)
                    fake_tracks.last_output = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.long_memory = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    active_tracks = fake_tracks
                tracks.append(active_tracks)
        else:
            # Eval only has B=1.
            assert len(previous_tracks) == 1 and len(new_tracks) == 1
            new_tracks[0].last_output = new_tracks[0].output_embed
            new_tracks[0].long_memory = new_tracks[0].query_embed
            active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[0], new_tracks[0])
            active_tracks = active_tracks[active_tracks.ids >= 0]
            tracks.append(active_tracks)
        return tracks


def build(config: dict):
    return QueryUpdater(
            hidden_dim=config["HIDDEN_DIM"],
            ffn_dim=config["FFN_DIM"],
            dropout=config["DROPOUT"],
            tp_drop_ratio=config["TP_DROP_RATE"] if "TP_DROP_RATE" in config else 0.0,
            fp_insert_ratio=config["FP_INSERT_RATE"] if "FP_INSERT_RATE" in config else 0.0,
            use_checkpoint=config["USE_CHECKPOINT"],
            use_dab=config["USE_DAB"],
            update_threshold=config["UPDATE_THRESH"],
            long_memory_lambda=config["LONG_MEMORY_LAMBDA"],
            visualize=config["VISUALIZE"]
        )

