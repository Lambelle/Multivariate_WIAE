import torch

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Custom_Dataset(Dataset):
    def __init__(
        self,
        input_size: int,
        data_path: str,
        dataset: str,
        flag: str,
        pred_step,
    ):
        self.input_size = input_size
        self.train_data, self.test_data = eval("prepare_" + dataset)(data_path)
        self.flag = flag
        self.pred_step = pred_step

    def __len__(self):
        if self.flag == "train":
            return self.train_data.shape[1] - self.input_size
        else:
            return self.test_data.shape[1] - self.input_size

    def __getitem__(self, index):
        if self.flag == "train":
            y_input = self.train_data[:, index : index + self.input_size].clone()

            if torch.all(y_input.std(dim=1) > 0):
                mean = y_input.mean(dim=1)
                std = y_input.std(dim=1)
                y_input = (y_input - mean.unsqueeze(1)) / std.unsqueeze(1)

            return y_input.squeeze(1)

        if self.flag == "test":
            y_input = self.test_data[:, index : index + self.input_size].clone()
            if torch.all(y_input.std(dim=1)) > 0:
                mean = y_input.mean(dim=1)
                std = y_input.std(dim=1)
                y_input = (y_input - mean.unsqueeze(1)) / std.unsqueeze(1)

            y_true = self.test_data[0,-self.pred_step:]  # Only return the channel that needs prediction, which is always placed as the first channel
            return y_input.squeeze(1), y_true


def prepare_PJM(csv_path: str):
    train_start = "2022-01-01 00:00:00"
    train_end = "2022-09-01 23:00:00"
    test_start = "2022-09-01 00:00:00"
    test_end = "2022-12-31 00:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna("bfill", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)
