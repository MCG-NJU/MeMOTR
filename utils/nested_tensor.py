# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
# @Description  : NestedTensor，modified from MOTR/Deformable DETR
import torch

from typing import Optional, List


class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, masks: Optional[torch.Tensor]):
        """
        Args:
            tensors: Tensor, (B, C, H, W)
            masks: Tensor, (B, H, W)
        """
        assert tensors.shape[0] == masks.shape[0], \
            f"tensors have batch size {tensors.shape[0]} but get {masks.shape[0]} for mask."
        self.tensors = tensors
        self.masks = masks

    def to(self, device, non_blocking=False):
        """
        Args:
            device:
            non_blocking:
        """
        tensors = self.tensors.to(device, non_blocking=non_blocking)
        if self.masks is None:
            masks = None
        else:
            masks = self.masks.to(device, non_blocking=non_blocking)
        return NestedTensor(tensors=tensors, masks=masks)

    def decompose(self) -> [torch.Tensor, torch.Tensor]:
        return self.tensors, self.masks

    def __repr__(self):
        return str(self.tensors)


def tensor_list_to_nested_tensor(tensor_list: List[torch.Tensor], size_divisibility: int = 32) -> NestedTensor:
    assert tensor_list[0].dim() == 3, f"Tensor should have 3 dimensions, but get {tensor_list[0].dim()}"
    heights, widths = zip(*[t.shape[1:] for t in tensor_list])
    final_shape = [len(tensor_list)] + [tensor_list[0].shape[0]] + list(map(max, (heights, widths)))
    final_b, final_c, final_h, final_w = final_shape
    if size_divisibility > 0:
        stride = size_divisibility
        final_h = (final_h + (stride - 1)) // stride * stride
        final_w = (final_w + (stride - 1)) // stride * stride
    final_shape = [final_b, final_c, final_h, final_w]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensors = torch.zeros(final_shape, dtype=dtype, device=device)
    masks = torch.ones((final_b, final_h, final_w), dtype=torch.bool, device=device)
    for input_tensor, pad_tensor, mask in zip(tensor_list, tensors, masks):
        assert input_tensor.shape[0] == final_shape[1], "Tensor channel size should be equal."
        pad_tensor[: input_tensor.shape[0], : input_tensor.shape[1], : input_tensor.shape[2]].copy_(input_tensor)
        mask[: input_tensor.shape[1], : input_tensor.shape[2]] = False
    return NestedTensor(tensors=tensors, masks=masks)

