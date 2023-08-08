# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
# @Description  : 用于 backbone 的设计和搭建
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from utils.nested_tensor import NestedTensor

from .position_embedding import build as build_position_embedding


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    Code reference: https://github.com/megvii-research/MOTR/blob/main/models/backbone.py
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """
    ResNet with frozen BatchNorm as backbone.
    """
    def __init__(self, backbone_name: str, train_backbone: bool, return_interm_layers: bool):
        """
        初始化一个 Backbone

        Args:
            backbone_name: backbone_name, only resnet50 is supported.
            train_backbone: whether finetune this backbone.
            return_interm_layers: whether return the intermediate layers' outputs.
        """
        super(Backbone, self).__init__()
        assert backbone_name == "resnet50", f"Backbone do not support '{backbone_name}'."
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)

        for name, parameter in backbone.named_parameters():
            if not train_backbone or ("layer2" not in name and "layer3" not in name and "layer4" not in name):
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {
                "layer2": "0",
                "layer3": "1",
                "layer4": "2"
            }
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4", "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, ntensor: NestedTensor):
        outputs = self.backbone(ntensor.tensors)
        res: Dict[str, NestedTensor] = dict()
        for name, output in outputs.items():
            masks = ntensor.masks
            assert masks is not None, "Masks should be NOT NONE."
            masks = F.interpolate(masks[None].float(), mode="nearest", size=output.shape[-2:]).to(masks.dtype)[0]
            res[name] = NestedTensor(output, masks)
        return res


class BackboneWithPE(nn.Module):
    """
    Backbone with Position Embedding.
    Input: NestedTensor in (B, C, H, W)
    Output: Multi layer (B, C, H, W) as Image Features, multi layer (B, 2*num_pos_feats, H, W) as Position Embedding.
    """
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super(BackboneWithPE, self).__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, ntensor: NestedTensor) -> (List[NestedTensor], List[torch.Tensor]):
        backbone_outputs = self.backbone(ntensor)
        features: List[NestedTensor] = list()
        pos_embeds: List[torch.Tensor] = list()
        # Image Features
        for _, output in sorted(backbone_outputs.items()):
            features.append(output)
        # Position Embedding
        for feature in features:
            pos_embeds.append(self.position_embedding(feature))

        return features, pos_embeds     # (B, C, H, W), (B, 2*num_pos_feats, H, W)，C is different in different layers.

    def n_inter_layers(self):
        return len(self.strides)

    def n_inter_channels(self):
        return self.num_channels


def build(config: dict) -> BackboneWithPE:
    position_embedding = build_position_embedding(config=config)
    backbone = Backbone(backbone_name=config["BACKBONE"], train_backbone=True, return_interm_layers=True)
    return BackboneWithPE(backbone=backbone, position_embedding=position_embedding)
