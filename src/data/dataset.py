"""Read the benchmark data and read or write the predictions of a model, as NumPy arrays.

All the functions expect to be called from the main folder of the repository.
"""

import os

import h5py
import numpy as np

DATA_FOLDER = "data"
RESULTS_FOLDER = "results"
FIRST_YEAR = 1968  # the times are in years since 1 January 1968


def available_splits():
    """The test file is not public: only the splits found in the data folder are returned."""
    splits = []
    for split in ["train", "validation", "test"]:
        if os.path.exists(f"{DATA_FOLDER}/{split}.h5"):
            splits.append(split)
    return splits


def load_split(split):
    """Data of "train", "validation" or "test".

    parameters: (n_scenarios, 9)
    times:      (n_times,), in years since 1 January 1968
    cff:        (n_scenarios, n_times, n_cells), in MPa. None if the file has no CFF.
    """
    with h5py.File(f"{DATA_FOLDER}/{split}.h5", "r") as file:
        parameters = file["parameters"][:]
        times = file["times"][:]
        cff = None
        if "cff" in file:
            cff = file["cff"][:]
    return parameters, times, cff


def load_samples(split):
    """Data of a split as input-output pairs, one for each scenario and time.

    inputs: (n_samples, 10), the 9 parameters followed by the time
    cff:    (n_samples, n_cells), in MPa. None if the file has no CFF.
    """
    parameters, times, cff = load_split(split)
    inputs = []
    for scenario_parameters in parameters:
        for time in times:
            inputs.append([*scenario_parameters, time])
    inputs = np.array(inputs)
    if cff is not None:
        cff = cff.reshape(len(inputs), -1)
    return inputs, cff


def save_predictions(split, cff):
    """Save the CFF predicted for the inputs of load_samples(split), in MPa."""
    parameters, times, _ = load_split(split)
    cff = np.asarray(cff).reshape(len(parameters), len(times), -1)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    with h5py.File(f"{RESULTS_FOLDER}/predictions_{split}.h5", "w") as file:
        file["cff"] = cff


def load_predictions(split):
    """Predicted CFF with the same shape as the CFF of load_split(split)."""
    with h5py.File(f"{RESULTS_FOLDER}/predictions_{split}.h5", "r") as file:
        return file["cff"][:]


def load_fault_mesh():
    """Triangular mesh of the fault: one triangle for each value of the CFF.

    nodes:     (n_nodes, 3), coordinates in metres
    triangles: (n_cells, 3), indices of the three nodes of each cell
    """
    with h5py.File(f"{DATA_FOLDER}/fault_mesh.h5", "r") as file:
        return file["nodes"][:], file["triangles"][:]
