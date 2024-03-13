# @Author       : Ruopeng Gao
# @Date         : 2023/3/10
# Modified from ./dancetrack.py
import os
import json
from math import floor
from random import randint

import torch
from PIL import Image
import data.transforms as T
# from typing import List
# from torch.utils.data import Dataset
from .mot import MOTDataset
from collections import defaultdict

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


def category_to_label():
    return {
        "pedestrian": 0,
        "rider": 1,
        "car": 2,
        "truck": 3,
        "bus": 4,
        "train": 5,
        "motorcycle": 6,
        "bicycle": 7,
        "other vehicle": 2,
        "other person": 0,
        "trailer": 3
    }

def label_to_category():
    return {
        0: "pedestrian",
        1: "rider",
        2: "car",
        3: "truck",
        4: "bus",
        5: "train",
        6: "motorcycle",
        7: "bicycle"
    }


class BDD100K(MOTDataset):
    def __init__(self, config: dict, split: str, transform):
        """
        Args:
            config:
            split:
            transform:
        """
        super(BDD100K, self).__init__(config=config, split=split, transform=transform)

        self.config = config        # 配置信息
        self.transform = transform  # 数据需要经过的变换
        assert split == "train", f"Split {split} is not supported!"
        self.images_dir = os.path.join(config["DATA_ROOT"], "bdd100k", "images/track/train/")
        self.gts_dir = os.path.join(config["DATA_ROOT"], "bdd100k", "filter_labels/track/train/")
        assert os.path.exists(self.images_dir), f"Dir {self.images_dir} is not exist."

        # 采样的逻辑：
        self.sample_steps: list = config["SAMPLE_STEPS"]
        self.sample_intervals: list = config["SAMPLE_INTERVALS"]
        self.sample_modes: list = config["SAMPLE_MODES"]
        self.sample_lengths: list = config["SAMPLE_LENGTHS"]
        # 当前的采样策略，随着 epoch 的迭代，如下的内容应该会发生变化
        self.sample_stage = None
        self.sample_begin_frames = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.gts = defaultdict(lambda: defaultdict(list))

        for vid in os.listdir(self.images_dir):
            frame_names = os.listdir(os.path.join(self.images_dir, vid))
            frame_names.sort()
            for frame_name in frame_names:
                gt_name = frame_name.replace(".jpg", ".txt")
                gt_path = os.path.join(self.gts_dir, vid, gt_name)
                t = int(gt_name[:-4].split("-")[-1])
                if os.path.exists(gt_path):
                    for line in open(gt_path):
                        c, i, *xywh = line[:-1].split(" ")
                        c, i = map(int, (c, i))
                        x, y, w, h = map(float, xywh)
                        self.gts[vid][t].append([c, i, x, y, w, h])

        self.set_epoch(0)   # init for each epoch
        # Default: 275030 length

        return

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        imgs, infos = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos)
        return {
            "imgs": imgs,
            "infos": infos
        }

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frames)

    def sample_frames_idx(self, vid: int, begin_frame: int) -> list[int]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idxs = [begin_frame + interval * i for i in range(self.sample_length)]
            # lack in BDD100K
            is_lack = False
            for _ in frame_idxs:
                if _ not in self.gts[vid]:
                    is_lack = True
                    break
            if is_lack:
                frame_idxs = [begin_frame + _ for _ in range(self.sample_length)]
            return frame_idxs
        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch: int):
        self.sample_begin_frames = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        for vid in self.gts.keys():
            t_min = min(self.gts[vid].keys())
            t_max = max(self.gts[vid].keys())
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                filter_out = False
                for _ in range(self.sample_length):
                    if t + _ not in self.gts[vid]:
                        filter_out = True
                        break
                if filter_out is False:
                    self.sample_begin_frames.append((vid, t))

        return

    def get_single_frame(self, vid: str, idx: int):
        img_path = os.path.join(self.images_dir, vid, f"{vid}-{idx:07d}.jpg")
        img = Image.open(img_path)
        info = {}

        info["boxes"] = list()
        info["ids"] = list()
        info["labels"] = list()
        info["areas"] = list()

        info["frame_idx"] = torch.as_tensor(idx)
        for label, i, *xywh in self.gts[vid][idx]:
            info["boxes"].append(list(map(float, xywh)))
            info["areas"].append(xywh[2] * xywh[3])     # area = w * h
            info["ids"].append(i)
            info["labels"].append(label - 1)            # different category in BDD100K

        # fake GTs, a hack implementation
        if len(info["ids"]) == 0:
            info["boxes"].append([0.5, 0.5, 0.5, 0.5])
            info["areas"].append(0.0)
            info["ids"].append(0)
            info["labels"].append(0)

        info["boxes"] = torch.as_tensor(info["boxes"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["ids"] = torch.as_tensor(info["ids"], dtype=torch.long)
        info["labels"] = torch.as_tensor(info["labels"], dtype=torch.long)

        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]    # xywh to cxcywh
        else:
            pass

        return img, info

    def get_multi_frames(self, vid: str, idxs: list[int]):
        return zip(*[self.get_single_frame(vid=vid, idx=i) for i in idxs])


def transfroms_for_train():
    # scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]  # from MOTR
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]  # from COCO
    # NOTE: For BDD100K, we use 1333 instead of 1536, as the max size, because we do not have enough time to train.
    #       At that time, we need to provide exp results within 7 days.
    #       Therefore, I'm not sure whether the performance will be better if we use 1536 as the max size.
    return T.MultiCompose([
        T.MultiRandomHorizontalFlip(),
        T.MultiRandomSelect(
            # T.MultiRandomResize(sizes=scales, max_size=1536),
            T.MultiRandomResize(sizes=scales, max_size=1333),
            T.MultiCompose([
                # T.MultiRandomResize([800, 1000, 1200]),
                T.MultiRandomResize([400, 500, 600]),
                # T.MultiRandomCrop(min_size=384, max_size=600, overflow_bbox=False),
                T.MultiRandomCrop(min_size=384, max_size=600, overflow_bbox=True),
                # T.MultiRandomResize(sizes=scales, max_size=1536)
                T.MultiRandomResize(sizes=scales, max_size=1333)
            ])
        ),
        T.MultiHSV(),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # from COCO/MOTR
        ])
    ])


def build(config: dict, split: str):
    if split == "train":
        return BDD100K(config=config, split=split, transform=transfroms_for_train())
    else:
        raise ValueError(f"Data split {split} is not supported for DanceTrack dataset.")
