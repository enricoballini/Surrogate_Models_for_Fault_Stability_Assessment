"""Compute the metrics of the predictions saved in the results folder.

The RMSE of DCFF is computed for each split, separately for the times of the training
set and for the future times. The results are printed and saved in results/metrics.json.
"""

import json
import os

import numpy as np

from src.metrics.metrics import (
    future_times,
    load_references_and_predictions,
    rmse,
    rmse_per_field,
)

metrics = {}

for split, evaluation in load_references_and_predictions().items():
    times, cff_reference, cff_predicted = evaluation
    future = future_times(times)
    groups = {"training times": ~future}
    if future.any():
        groups["future times"] = future

    for group, selected_times in groups.items():
        reference = cff_reference[:, selected_times]
        predicted = cff_predicted[:, selected_times]
        errors = rmse_per_field(predicted, reference)
        metrics[f"{split}, {group}"] = {
            "rmse": float(rmse(predicted, reference)),
            "rmse_per_field_p10": float(np.percentile(errors, 10)),
            "rmse_per_field_p50": float(np.percentile(errors, 50)),
            "rmse_per_field_p90": float(np.percentile(errors, 90)),
        }

print("RMSE of DCFF = CFF_predicted - CFF_reference, in MPa")
print(f"{'':28s}{'all values':>12s}{'P10':>8s}{'P50':>8s}{'P90':>8s}   (P: per field)")
for name, values in metrics.items():
    print(
        f"{name:28s}{values['rmse']:12.3f}{values['rmse_per_field_p10']:8.3f}"
        f"{values['rmse_per_field_p50']:8.3f}{values['rmse_per_field_p90']:8.3f}"
    )

if os.path.exists("results/efficiency.json"):
    with open("results/efficiency.json") as file:
        metrics["efficiency"] = json.load(file)
    print("efficiency:", metrics["efficiency"])

with open("results/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)
