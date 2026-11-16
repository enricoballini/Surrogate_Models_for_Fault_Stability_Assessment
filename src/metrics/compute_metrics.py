"""Compute the metrics of the predictions saved in the results folder.

The RMSE of DCFF is computed for each split, separately for the times of the training
set and for the future times. The results are printed and saved in results/metrics.json.
"""

import json
import os

from src.metrics.metrics import future_times, load_references_and_predictions, rmse

metrics = {"rmse": {}}

for split, evaluation in load_references_and_predictions().items():
    times, cff_reference, cff_predicted = evaluation
    future = future_times(times)
    groups = {"training times": ~future}
    if future.any():
        groups["future times"] = future

    for group, selected_times in groups.items():
        reference = cff_reference[:, selected_times]
        predicted = cff_predicted[:, selected_times]
        metrics["rmse"][f"{split}, {group}"] = float(rmse(predicted, reference))

print("RMSE of DCFF = CFF_predicted - CFF_reference, in MPa")
for name, value in metrics["rmse"].items():
    print(f"{name:28s}{value:8.3f}")

if os.path.exists("results/efficiency.json"):
    with open("results/efficiency.json") as file:
        metrics["efficiency"] = json.load(file)
    print("efficiency:", metrics["efficiency"])

with open("results/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)
