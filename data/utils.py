import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from typing import Tuple, Any, Union, Type, Dict, List
from collections import defaultdict


def collate_fn(batch):
    collated_batch = defaultdict(list)
    for data in batch:
        collated_batch["imgs"].append(data["imgs"])
        collated_batch["infos"].append(data["infos"])
    return collated_batch
