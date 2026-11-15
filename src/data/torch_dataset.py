"""Benchmark data for PyTorch users."""

import torch
from torch.utils.data import Dataset

from src.data.dataset import load_samples


class FaultStabilityDataset(Dataset):
    """Samples of "train" or "validation".

    Each sample is (inputs, cff):
    inputs: the 9 parameters followed by the time
    cff:    the CFF on the 416 fault cells, in MPa
    """

    def __init__(self, split):
        inputs, cff = load_samples(split)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.cff = torch.tensor(cff, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.cff[index]
