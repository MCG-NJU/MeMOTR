# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from os import path
from typing import List
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from models import build_model
from models.utils import load_checkpoint, get_model
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict, is_distributed, distributed_world_size, distributed_rank, inverse_sigmoid
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from log.logger import Logger
from data.seq_dataset import SeqDataset
from structures.track_instances import TrackInstances


class Submitter:
    def __init__(self, dataset_name: str, split_dir: str, seq_name: str, outputs_dir: str, model: nn.Module,
                 det_score_thresh: float = 0.7, track_score_thresh: float = 0.6, result_score_thresh: float = 0.7,
                 miss_tolerance: int = 5,
                 use_motion: bool = False, motion_lambda: float = 0.5,
                 motion_min_length: int = 3, motion_max_length: int = 5,
                 use_dab: bool = False,
                 visualize: bool = False):
        self.dataset_name = dataset_name
        self.seq_name = seq_name
        self.seq_dir = path.join(split_dir, seq_name)
        self.outputs_dir = outputs_dir
        self.predict_dir = path.join(self.outputs_dir, "tracker")
        self.model = model
        self.tracker = RuntimeTracker(det_score_thresh=det_score_thresh, track_score_thresh=track_score_thresh,
                                      miss_tolerance=miss_tolerance,
                                      use_motion=use_motion,
                                      motion_min_length=motion_min_length, motion_max_length=motion_max_length,
                                      visualize=visualize, use_dab=use_dab)
        self.result_score_thresh = result_score_thresh
        self.motion_lambda = motion_lambda
        self.dataset = SeqDataset(seq_dir=self.seq_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=4, shuffle=False)
        self.device = next(self.model.parameters()).device
        self.use_dab = use_dab
        self.use_motion = use_motion
        self.visualize = visualize
        # 对路径进行一些操作
        os.makedirs(self.predict_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_dir, f'{self.seq_name}.txt')):
            os.remove(os.path.join(self.predict_dir, f'{self.seq_name}.txt'))
        self.model.eval()
        return

    @torch.no_grad()
    def run(self):
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 use_dab=self.use_dab).to(self.device)]
        bdd100k_results = []    # for bdd100k, will be converted into json file, different from other datasets.
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # image: (1, C, H, W); ori_image: (1, H, W, C)
            frame = tensor_list_to_nested_tensor([image[0]]).to(self.device)
            res = self.model(frame=frame, tracks=tracks)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)

            # We do not use this...
            # but I do not want to remove this part.
            # WHAT IF it breaks down!!!
            # of course not :)
            if self.use_motion:
                for _ in range(len(tracks[0])):
                    if tracks[0].disappear_time[_].item() > 0:
                        if len(self.tracker.motions[tracks[0].ids[_].item()]) >= \
                               self.tracker.motions[tracks[0].ids[_].item()].min_record_length:
                            tracks[0].ref_pts[_] = inverse_sigmoid(
                                tracks[0].last_appear_boxes[_]
                            ) + self.motion_lambda * self.tracker.motions[tracks[0].ids[_].item()].get_box_delta(
                                miss_length=tracks[0].disappear_time[_].item()
                            ).to(tracks[0].last_appear_boxes.device)

            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
            # box = [x, y, w, h]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            # to xyxy:
            tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))
            if self.dataset_name == "BDD100K":
                self.update_results(tracks_result=tracks_result, frame_idx=i, results=bdd100k_results, img_path=info[0])
            else:
                self.write_results(tracks_result=tracks_result, frame_idx=i)

            if self.visualize:
                os.makedirs(f"./outputs/visualize_tmp/frame_{i+1}/", exist_ok=False)
                os.system(f"mv ./outputs/visualize_tmp/query_updater/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/decoder/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/memotr/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/runtime_tracker/ ./outputs/visualize_tmp/frame_{i+1}/")

        if self.visualize:
            visualize_save_dir = os.path.join("./outputs/visualize/", self.seq_name)
            os.makedirs(visualize_save_dir, exist_ok=True)
            os.system(f"mv ./outputs/visualize_tmp/* {visualize_save_dir}")

        with open(os.path.join(self.predict_dir, '{}.json'.format(self.seq_name)), 'w', encoding='utf-8') as f:
            json.dump(bdd100k_results, f)

        return

    @staticmethod
    def filter_by_score(tracks: TrackInstances, thresh: float = 0.7):
        keep = torch.max(tracks.scores, dim=-1).values > thresh
        return tracks[keep]

    @staticmethod
    def filter_by_area(tracks: TrackInstances, thresh: int = 100):
        assert len(tracks.area) == len(tracks.ids), f"Tracks' 'area' should have the same dim with 'ids'"
        keep = tracks.area > thresh
        return tracks[keep]

    def update_results(self, tracks_result: TrackInstances, frame_idx: int, results: list, img_path: str):
        # Only be used for BDD100K:
        bdd_cls2label = {
            1: "pedestrian",
            2: "rider",
            3: "car",
            4: "truck",
            5: "bus",
            6: "train",
            7: "motorcycle",
            8: "bicycle"
        }
        frame_result = {
            "name": img_path.split("/")[-1],
            "videoName": img_path.split("/")[-1][:-12],
            # "frameIndex": int(img_path.split("/")[-1][:-4].split("-")[-1]) - 1
            "frameIndex": frame_idx,
            "labels": []
        }
        for i in range(len(tracks_result)):
            x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
            ID = str(tracks_result.ids[i].item())
            label = bdd_cls2label[tracks_result.labels[i].item() + 1]
            frame_result["labels"].append(
                {
                    "id": ID,
                    "category": label,
                    "box2d": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }
            )
        results.append(frame_result)
        return

    def write_results(self, tracks_result: TrackInstances, frame_idx: int):
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "a") as file:
            for i in range(len(tracks_result)):
                if self.dataset_name == "DanceTrack" or self.dataset_name == "SportsMOT" \
                        or self.dataset_name == "MOT17" or self.dataset_name == "MOT17_SPLIT":
                    x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
                    w, h = x2 - x1, y2 - y1
                    result_line = f"{frame_idx+1}," \
                                  f"{tracks_result.ids[i].item()}," \
                                  f"{x1},{y1},{w},{h},1,-1,-1,-1\n"
                else:
                    raise ValueError(f"{self.dataset_name} dataset is not supported for submit process.")
                file.write(result_line)
        return


def submit(config: dict):
    submit_logger = Logger(logdir=os.path.join(config["SUBMIT_DIR"], config["SUBMIT_DATA_SPLIT"]), only_main=True)
    submit_logger.show(head="Configs:", log=config)
    submit_logger.write(log=config, filename="config.yaml", mode="w")

    assert config["SUBMIT_DIR"] is not None, f"'--submit-dir' must not be None for submit process."
    assert config["SUBMIT_MODEL"] is not None, f"'--submit-model' must not be None for submit process."
    assert config["SUBMIT_DATA_SPLIT"] is not None, f"'--submit-data-split' must not be None for submit process."
    train_config = yaml_to_dict(path=path.join(config["SUBMIT_DIR"], "train/config.yaml"))

    data_root = config["DATA_ROOT"]
    dataset_name = train_config["DATASET"]
    config["DATASET"] = dataset_name
    dataset_split = config["SUBMIT_DATA_SPLIT"]
    outputs_dir = path.join(config["SUBMIT_DIR"], dataset_split)
    use_dab = train_config["USE_DAB"]
    det_score_thresh = config["DET_SCORE_THRESH"]
    track_score_thresh = config["TRACK_SCORE_THRESH"]
    result_score_thresh = config["RESULT_SCORE_THRESH"]
    use_motion = config["USE_MOTION"]
    motion_min_length = config["MOTION_MIN_LENGTH"]
    motion_max_length = config["MOTION_MAX_LENGTH"]
    motion_lambda = config["MOTION_LAMBDA"]
    miss_tolerance = config["MISS_TOLERANCE"]

    model = build_model(config=train_config)
    load_checkpoint(
        model=model,
        path=path.join(config["SUBMIT_DIR"], config["SUBMIT_MODEL"])
    )
    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        data_split_dir = path.join(data_root, dataset_name, dataset_split)
    elif dataset_name == "BDD100K":
        data_split_dir = path.join(data_root, dataset_name, "images/track/", dataset_split)
    else:
        data_split_dir = path.join(data_root, dataset_name, "images", dataset_split)
    seq_names = os.listdir(data_split_dir)

    if is_distributed():
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=False)
        total_seq_names = seq_names
        seq_names = []
        for i in range(len(total_seq_names)):
            if i % distributed_world_size() == distributed_rank():
                seq_names.append(total_seq_names[i])

    for seq_name in seq_names:
        seq_name = str(seq_name)
        submitter = Submitter(
            dataset_name=dataset_name,
            split_dir=data_split_dir,
            seq_name=seq_name,
            outputs_dir=outputs_dir,
            model=model,
            use_dab=use_dab,
            det_score_thresh=det_score_thresh,
            track_score_thresh=track_score_thresh,
            result_score_thresh=result_score_thresh,
            use_motion=use_motion,
            motion_min_length=motion_min_length,
            motion_max_length=motion_max_length,
            motion_lambda=motion_lambda,
            miss_tolerance=miss_tolerance
        )
        submitter.run()
    return
