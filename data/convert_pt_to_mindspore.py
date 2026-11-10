"""One-time converter from trusted legacy PyTorch datasets to NumPy pickle.

PyTorch is required only while running this converter. MindSpore training does
not import or depend on PyTorch after conversion.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


class ListDataset:
    def __init__(self, *datalist):
        self.datalist = datalist

    def __getitem__(self, index):
        return tuple(data[index] for data in self.datalist)

    def __len__(self):
        return len(self.datalist[0])


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return [as_numpy(item) for item in value]
    return value


def load_legacy(path):
    main_module = sys.modules["__main__"]
    setattr(main_module, "ListDataset", ListDataset)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    for split in ("train", "valid", "test"):
        source = args.dataset_dir / f"{split}_sets_EMBSR.pt"
        destination = args.dataset_dir / f"{split}_sets_mindspore.pkl"
        dataset = load_legacy(source)
        converted = [tuple(as_numpy(field) for field in dataset[index]) for index in range(len(dataset))]
        with destination.open("wb") as stream:
            pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"converted {len(converted)} examples: {destination}")


if __name__ == "__main__":
    main()
