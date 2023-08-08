# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.
import os
import json
import argparse
import yaml

from tqdm import tqdm
from typing import List, Any
from torch.utils import tensorboard as tb

from log.log import MetricLog
from utils.utils import is_main_process


class ProgressLogger:
    def __init__(self, total_len: int, head: str = None, only_main: bool = True):
        self.only_main = only_main
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.total_len = total_len
            self.tqdm = tqdm(total=total_len)
            self.head = head
        else:
            self.total_len = None
            self.tqdm = None
            self.head = None

    def update(self, step_len: int, **kwargs: Any):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.tqdm.set_description(self.head)
            self.tqdm.set_postfix(**kwargs)
            self.tqdm.update(step_len)
        else:
            return


class Logger:
    """
    Log information.
    """
    def __init__(self, logdir: str, only_main: bool = True):
        self.only_main = only_main
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.logdir = logdir
            os.makedirs(self.logdir, exist_ok=True)
            # os.makedirs(os.path.join(self.logdir, "tb_log"), exist_ok=True)
            self.tb_iters_logger: tb.SummaryWriter = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_iters_log"))
            self.tb_epochs_logger: tb.SummaryWriter = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_epochs_log"))
        else:
            self.logdir = None
            self.tb_iters_logger: tb.SummaryWriter | None = None
            self.tb_epochs_logger: tb.SummaryWriter | None = None
        return

    def show(self, head: str = "", log: str | dict | MetricLog = ""):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            print(f"{head} {log}")
        else:
            pass
        return

    def write(self, head: str = "", log: dict | str | MetricLog = "", filename: str = "log.txt", mode: str = "a"):
        """
        Logger write a log to a file.

        Args:
            head: Log head like self.show.
            log: A log.
            filename: Write file name.
            mode: Open file with this mode.
        """
        if (self.only_main and is_main_process()) or (self.only_main is False):
            if isinstance(log, dict):
                if head != "":
                    raise Warning("Log is a dict, Do not support 'head' attr.")
                if len(filename) > 5 and filename[-5:] == ".yaml":
                    self.write_dict_to_yaml(log, filename, mode)
                elif len(filename) > 5 and filename[-5:] == ".json":
                    self.write_dict_to_json(log, filename, mode)
                elif len(filename) > 4 and filename[-4:] == ".txt":
                    self.write_dict_to_json(log, filename, mode)
                else:
                    raise RuntimeError("Filename '%s' is not supported for dict log." % filename)
            elif isinstance(log, MetricLog):
                with open(os.path.join(self.logdir, filename), mode=mode) as f:
                    f.write(f"{head} {log}\n")
            elif isinstance(log, str):
                with open(os.path.join(self.logdir, filename), mode=mode) as f:
                    f.write(f"{head} {log}\n")
            else:
                raise RuntimeError("Log type '%s' is not supported." % type(log))
        else:
            pass
        return

    def write_dict_to_yaml(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .yaml file.

        Args:
            log: A dict log.
            filename: A yaml file's name.
            mode: Open with this mode.
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            yaml.dump(log, f, allow_unicode=True)
        return

    def write_dict_to_json(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .json file.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w" or "a".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
        return

    def tb_add_scalar(self, tag: str, scalar_value: float, global_step: int, mode: str):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            if mode == "iters":
                writer: tb.SummaryWriter = self.tb_iters_logger
            else:
                writer: tb.SummaryWriter = self.tb_epochs_logger
            writer.add_scalar(
                tag=tag,
                scalar_value=scalar_value,
                global_step=global_step
            )
        return

    def tb_add_metric_log(self, log: MetricLog, steps: int, mode: str):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            log_keys = log.metrics.keys()
            box_l1_loss_keys, box_giou_loss_keys, label_focal_loss_keys = [], [], []
            for k in log_keys:
                if "box_l1_loss" in k:
                    box_l1_loss_keys.append(k)  # like "frame0_box_l1_loss"
                elif "box_giou_loss" in k:
                    box_giou_loss_keys.append(k)
                elif "label_focal_loss" in k:
                    label_focal_loss_keys.append(k)
                else:
                    pass
            if mode == "iters":
                writer: tb.SummaryWriter = self.tb_iters_logger
            else:
                writer: tb.SummaryWriter = self.tb_epochs_logger
            writer.add_scalars(
                main_tag="box_l1_loss",
                tag_scalar_dict={k.split("_")[0]: log.metrics[k].avg if mode == "iters" else log.metrics[k].global_avg
                                 for k in box_l1_loss_keys},
                global_step=steps
            )
            writer.add_scalars(
                main_tag="box_giou_loss",
                tag_scalar_dict={k.split("_")[0]: log.metrics[k].avg if mode == "iters" else log.metrics[k].global_avg
                                 for k in box_giou_loss_keys},
                global_step=steps
            )
            writer.add_scalars(
                main_tag="label_focal_loss",
                tag_scalar_dict={k.split("_")[0]: log.metrics[k].avg if mode == "iters" else log.metrics[k].global_avg
                                 for k in label_focal_loss_keys},
                global_step=steps
            )

            if "total_loss" in log_keys:
                writer.add_scalar(
                    tag="loss",
                    scalar_value=log.metrics["total_loss"].avg
                    if mode == "iters" else log.metrics["total_loss"].global_avg,
                    global_step=steps
                )
        else:
            pass
        return

    def tb_add_git_version(self, git_version: str):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            git_version = "null" if git_version is None else git_version
            self.tb_iters_logger.add_text(tag="git_version", text_string=git_version)
            self.tb_epochs_logger.add_text(tag="git_version", text_string=git_version)
        else:
            pass
        return


def parser_to_dict(log: argparse.ArgumentParser) -> dict:
    opts_dict = dict()
    for k, v in vars(log).items():
        if v:
            opts_dict[k] = v
    return opts_dict


