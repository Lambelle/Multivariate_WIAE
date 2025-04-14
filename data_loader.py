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
        input_dim: int,
        filter_size=1,
    ):
        self.input_size = input_size
        self.train_data, self.test_data = eval("prepare_" + dataset)(data_path)
        self.flag = flag
        self.input_dim = input_dim
        self.filter_size = filter_size
        self.pred_step = input_dim - 2 * (self.filter_size - 1)

    def __len__(self):
        if self.flag == "train":
            return self.train_data.shape[1] - self.input_size
        elif self.flag == "test":
            return (
                self.test_data.shape[1] - self.input_size - self.pred_step
            )  # Calculate the number of windows

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
            else:
                std = torch.ones(y_input.std(dim=1).shape)
                mean = torch.zeros(y_input.mean(dim=1).shape)

            y_true = self.test_data[
                :, index + self.input_size + self.pred_step
            ]  # Only return the channel that needs prediction, which is always placed as the first channel
            return y_input.squeeze(1), y_true, mean, std


def prepare_PJM(csv_path: str):
    train_start = "2022-01-01 00:00:00"
    train_end = "2022-09-01 23:00:00"
    test_start = "2022-09-01 00:00:00"
    test_end = "2022-12-31 00:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_PJM_spread(csv_path: str):
    train_start = "2022-01-01 00:00:00"
    train_end = "2022-09-01 23:00:00"
    test_start = "2022-09-01 00:00:00"
    test_end = "2022-12-31 00:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_spread(csv_path: str):
    train_start = "2023-07-01 00:00:00"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 23:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_RT(csv_path: str):
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 22:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_spread_2D(csv_path: str):
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 22:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_RTDA_load(csv_path: str):
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 22:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_RTDA_load_2D(csv_path: str):
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 22:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_PJM_ACE(csv_path: str):
    train_start = "2024-01-24 05:40:00"
    train_end = "2024-01-25 16:40:00"
    test_start = "2024-01-25 16:40:15"
    test_end = "2024-01-25 20:40:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_CTS(csv_path: str):
    train_start = "2024-02-08 00:00:00"
    train_end = "2024-02-18 00:00:00"
    test_start = "2024-02-18 00:00:15"
    test_end = "2024-02-20 00:10:15"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_CTS_2D(csv_path: str):
    train_start = "2024-02-08 00:00:00"
    train_end = "2024-02-18 00:00:00"
    test_start = "2024-02-18 00:00:15"
    test_end = "2024-02-20 00:10:15"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)
