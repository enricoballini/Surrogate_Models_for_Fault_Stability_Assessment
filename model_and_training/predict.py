"""Predict the CFF with the trained DL-ROM, for every split found in the data folder.

The predictions are saved in the results folder, where the metrics and the plots read
them. The number of trainable parameters and the inference time are saved as well.
"""

import json
import time

import numpy as np
import torch

from dlrom import DLROM
from src.data.dataset import available_splits, load_samples, load_split, save_predictions
from src.data.torch_dataset import FaultStabilityDataset

N_TIMING_REPETITIONS = 100

training_data = FaultStabilityDataset("train")
model = DLROM(training_data.inputs, training_data.cff)
model.load_state_dict(torch.load("results/dlrom.pt"))

for split in available_splits():
    inputs, _ = load_samples(split)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        cff = model(inputs)
    save_predictions(split, cff.numpy())
    print("saved the predictions of the", split, "split")

# inference time: prediction of all the times of one scenario
_, times, _ = load_split("train")
inputs_of_one_scenario = training_data.inputs[: len(times)]
durations = []
for repetition in range(N_TIMING_REPETITIONS):
    start = time.perf_counter()
    with torch.no_grad():
        model(inputs_of_one_scenario)
    durations.append(time.perf_counter() - start)

efficiency = {
    "trainable_parameters": sum(weights.numel() for weights in model.parameters()),
    "inference_time_per_scenario_seconds": float(np.median(durations)),
}
with open("results/efficiency.json", "w") as file:
    json.dump(efficiency, file, indent=4)
