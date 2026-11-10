#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path
from typing import Literal, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Backend = Literal["numpy", "torch"]
LabelFormat = Literal["name", "index", "onehot"]


def _read_labels(
        csv_path: str | Path
) -> List[str]:
    """ Get labels ordered by zero-based NPZ row index """

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    labels_by_index: Dict[int, str] = dict()

    for row in rows:
        index = int(row["index"])
        label = row["label"].strip()
        labels_by_index[index] = label
    return [labels_by_index[i] for i in range(len(rows))]


def get_class_names(
        csv_path: str | Path = "gas_labels.csv"
) -> List[str]:
    """ Class index -> class name, in deterministic alphabetical order """
    return sorted(set(_read_labels(csv_path)))


def load_dataset(
        npz_path: str | Path = "daily52_data.npz",
        csv_path: str | Path = "daily52_meta.csv",
        backend: Backend = "numpy",
        label_format: LabelFormat = "index",
        **kwargs
) -> Dict:
    """Return {'X': (N, 18, 540), 'y': (N,) or (N, C)}.

    backend='numpy': X and y are NumPy arrays (including string labels).
    backend='torch': X and numeric y are torch.Tensor; name y is list[str].
    device is applicable to backend='torch' (e.g., 'cpu' or 'cuda').
    """

    with np.load(npz_path, allow_pickle=False) as archive:
        X = archive["X"]

    names = _read_labels(csv_path)
    classes = sorted(set(names))
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    indices = np.array([class_to_index[name] for name in names], dtype=np.int64)

    if label_format == "name":
        y = np.asarray(names, dtype=str)
    elif label_format == "index":
        y = indices
    else:
        y = np.eye(len(classes), dtype=np.float32)[indices]

    # Get numpy array
    if backend == "numpy":
        return {"X": X, "y": y}

    # Get pytorch tensor
    X_tensor = torch.from_numpy(np.ascontiguousarray(X))

    if label_format == "name":
        y_torch = names.copy()
    else:
        y_torch = torch.from_numpy(np.ascontiguousarray(y))

    return {"X": X_tensor, "y": y_torch}


if __name__ == "__main__":
    data = load_dataset(backend="numpy", label_format = "index")
    print(data['X'].shape, data['y'].shape, type(data['X']), type(data['y'][0]))

    data = load_dataset(backend="numpy", label_format = "name")
    print(data['X'].shape, data['y'].shape, type(data['X']), type(data['y'][0]))

    data = load_dataset(backend="torch", label_format="index")
    print(data['X'].shape, data['y'].shape, type(data['X']), type(data['y'][0]))

    data = load_dataset(backend="torch", label_format="name")
    print(data['X'].shape, len(data['y']), type(data['X']), type(data['y'][0]))
