"""Accuracy metrics of the benchmark, based on DCFF = CFF_predicted - CFF_reference."""

import os

import numpy as np

from src.data.dataset import available_splits, load_predictions, load_split

TOLERANCE = 0.1  # physically acceptable error on the CFF, in MPa


def rmse(cff_predicted, cff_reference):
    """RMSE of DCFF over all the values, in MPa."""
    dcff = cff_predicted - cff_reference
    return np.sqrt(np.mean(dcff**2))


def rmse_per_field(cff_predicted, cff_reference):
    """RMSE of DCFF over the fault cells, in MPa: one value for each scenario and time."""
    dcff = cff_predicted - cff_reference
    return np.sqrt(np.mean(dcff**2, axis=-1))


def future_times(times):
    """True for the times after the last time of the training set."""
    _, training_times, _ = load_split("train")
    return times > training_times.max()


def load_references_and_predictions():
    """Times, reference CFF and predicted CFF of every split that has both."""
    evaluations = {}
    for split in available_splits():
        _, times, cff_reference = load_split(split)
        has_predictions = os.path.exists(f"results/predictions_{split}.h5")
        if cff_reference is not None and has_predictions:
            evaluations[split] = (times, cff_reference, load_predictions(split))
    return evaluations
