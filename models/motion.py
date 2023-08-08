# We actually do not use this class.

import torch


class Motion:
    def __init__(self, min_record_length: int = 3, max_record_length: int = 5):
        self.boxes = torch.zeros((0, 4), dtype=torch.float)
        self.min_record_length = min_record_length
        self.max_record_length = max_record_length

    def add_box(self, box: torch.Tensor):
        self.boxes = torch.cat((self.boxes, box.reshape((1, 4))), dim=0)
        if self.boxes.shape[0] > self.max_record_length:
            self.boxes = self.boxes[-self.max_record_length:, :]

    def get_box_delta(self, miss_length: int):
        delta_sum = torch.zeros((1, 4), dtype=torch.float)
        for i in range(self.boxes.shape[0] - 1):
            delta_sum = delta_sum + (self.boxes[i+1] - self.boxes[i])
        delta = (miss_length / (self.boxes.shape[0] - 1)) * delta_sum
        return delta

    def __len__(self):
        return self.boxes.shape[0]

    def clear(self):
        self.boxes = torch.zeros((0, 4), dtype=torch.float)