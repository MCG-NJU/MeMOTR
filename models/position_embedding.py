# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import torch
import math
from torch import nn

from utils.nested_tensor import NestedTensor


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        if self.normalize is True and self.scale is None:
            raise ValueError("Scale should be NOT NONE when normalize is True.")
        if self.scale is not None and self.normalize is False:
            raise ValueError("Normalize should be True when scale is not None.")

    def forward(self, ntensor: NestedTensor) -> torch.Tensor:
        tensors, masks = ntensor.decompose()
        assert masks is not None, "Masks in ntensor should be NOT NONE."
        not_masks = ~masks
        y = not_masks.cumsum(dim=1, dtype=torch.float32)
        x = not_masks.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y = (y - 0.5) / (y[:, -1:, :] + eps) * self.scale
            x = (x - 0.5) / (x[:, :, -1:] + eps) * self.scale

        dim_i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensors.device)
        dim_i = self.temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / self.num_pos_feats)

        pos_x = x[:, :, :, None] / dim_i
        pos_y = y[:, :, :, None] / dim_i
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)    # (B, 2*num_pos_feats, H, W)
        return pos_embed


def build(config: dict):
    assert config["HIDDEN_DIM"] % 2 == 0, f"Hidden dim should be 2x, but get {config['HIDDEN_DIM']}."
    num_pos_feats = config["HIDDEN_DIM"] / 2
    return PositionEmbeddingSine(num_pos_feats=num_pos_feats, normalize=True, scale=2*math.pi, temperature=20)
