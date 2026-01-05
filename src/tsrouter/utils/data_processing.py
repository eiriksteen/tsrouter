import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class CLSData:
    x: np.ndarray
    y: np.ndarray
    # error lag is the error for all models on the current input window, meaning the pointwise SE for for the preds shifted by the seq_len
    preds: np.ndarray
    mse: pd.DataFrame
    error_lag: np.ndarray
    best_idx: pd.DataFrame
    id2label: dict
    name: str


dataset_paths = {
    "weather": Path.cwd() / "classification_data" / "weather",
    "ett": Path.cwd() / "classification_data" / "ETTh1",
    "exchange": Path.cwd() / "classification_data" / "Exchange",
}


def load_cls_data(
        dataset_name: Literal["weather", "ett", "exchange"],
        split: Literal["train", "test", "val"],
        include_mean=False,
        pred_len=96,
        models: Optional[list[str]] = None) -> CLSData:
    # The models are trained on a certain seq_len, but we can chop of predictions beyond a seq_len if we want to evaluate them on shorter horizons

    path = dataset_paths[dataset_name]
    x = np.load(path / f"{split}_inputs.npy")
    assert pred_len <= x.shape[1], f"seq_len {pred_len} is greater than the sequence length of the data {x.shape[1]}"
    y = np.load(path / f"{split}_outputs.npy")
    preds = np.load(path / f"{split}_predictions.npy")

    y = y[:, :pred_len]
    preds = preds[:, :, :pred_len]
    mse = ((y[None, :, :, :] - preds)**2).mean(-2)

    with open(path / "model_order.json") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    if include_mean:
        mean_preds = preds.mean(0)
        mean_mse = ((y - mean_preds)**2).mean(1)
        mse = np.concatenate((mse, mean_mse[None, :]))
        preds = np.concatenate((preds, mean_preds[None, :, :]))
        id2label[len(id2label)] = "Mean"

    best_idx = mse.argmin(0)

    # error lag
    preds_shifted = np.roll(preds, pred_len, axis=1)
    preds_shifted[:, :pred_len] = 0
    y_shifted = np.roll(y, pred_len, axis=0)  # [seq_len:]
    error_lag = ((preds_shifted - y_shifted)**2)
    # Set the first seq_len of error lag to 0 since they are not defined
    error_lag[:, :pred_len] = 0

    cls_data = CLSData(x, y, preds, mse, error_lag,
                       best_idx, id2label, f"{dataset_name}_{split}")

    if models is not None:
        cls_data = filter_cls_data_models(cls_data, models)

    return cls_data


def filter_cls_data_models(cls_data: CLSData, models: list[str]) -> CLSData:
    idx = [k for k, v in cls_data.id2label.items() if v in models]
    preds_filtered = cls_data.preds[idx]
    mse_filtered = cls_data.mse[idx]
    error_lag_filtered = cls_data.error_lag[idx]
    best_idx_filtered = mse_filtered.argmin(axis=0)
    id2label_filtered = {
        i: cls_data.id2label[old_idx] for i, old_idx in enumerate(idx)}

    return CLSData(
        x=cls_data.x,
        y=cls_data.y,
        preds=preds_filtered,
        mse=mse_filtered,
        error_lag=error_lag_filtered,
        best_idx=best_idx_filtered,
        id2label=id2label_filtered,
        name=cls_data.name
    )


def add_model_to_cls_data(preds: np.ndarray, model_name: str, cls_data: CLSData) -> CLSData:
    new_preds = np.concatenate((preds[None, :, :], cls_data.preds), axis=0)

    new_model_mse = ((cls_data.y - preds) ** 2).mean(axis=1)
    new_mse = np.concatenate((new_model_mse[None, :], cls_data.mse), axis=0)

    new_best_idx = new_mse.argmin(axis=0)

    # Since adding Oracle post-hoc, error_lag has no real meaning
    # Just pad with zeros to keep shape consistent
    zero_lag = np.zeros((1, *cls_data.error_lag.shape[1:]))
    new_error_lag = np.concatenate((zero_lag, cls_data.error_lag), axis=0)

    new_id2label = {k + 1: v for k, v in cls_data.id2label.items()}
    new_id2label[0] = model_name

    return CLSData(
        cls_data.x,
        cls_data.y,
        new_preds,
        new_mse,
        new_error_lag,
        new_best_idx,
        new_id2label,
        f"{cls_data.name} + {model_name}"
    )


def select_preds_from_idx(idx: np.ndarray, preds: np.ndarray):
    _, _, n_time, _ = preds.shape
    idx_reshaped = idx[np.newaxis, :, np.newaxis, :]
    A_broadcastable = np.repeat(idx_reshaped, n_time, axis=2)
    result_take = np.take_along_axis(preds, A_broadcastable, axis=0)
    output = np.squeeze(result_take, axis=0)
    return output


class CLSTorchDataset(Dataset):
    def __init__(self, cls_data: CLSData):
        self.cls_data = cls_data
        self.n_models, self.n_slices, self.seq_len, self.n_feats = cls_data.preds.shape

    def __len__(self):
        return self.n_slices

    def __getitem__(self, idx: int):
        x = self.cls_data.x[idx].T
        y = self.cls_data.y[idx].T
        preds = self.cls_data.preds[:, idx].transpose(0, 2, 1)
        mse = self.cls_data.mse[:, idx]
        relevance = self.cls_data.mse[:, idx, :].shape[0] - \
            self.cls_data.mse[:, idx, :].argsort(0).argsort(0)
        error_lag = self.cls_data.error_lag[:, idx].transpose(0, 2, 1)  # .T

        return {
            "x": x,
            "y": y,
            "preds": preds,
            "mse": mse,
            "relevance": relevance,
            "error_lag": error_lag
        }
