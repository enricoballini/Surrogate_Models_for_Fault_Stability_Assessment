"""Box plot of the RMSE of DCFF: one box for each split, one value for each field."""

import matplotlib.pyplot as plt

from src.metrics.metrics import (
    TOLERANCE,
    future_times,
    load_references_and_predictions,
    rmse_per_field,
)

errors = []
labels = []

for split, evaluation in load_references_and_predictions().items():
    times, cff_reference, cff_predicted = evaluation
    future = future_times(times)
    errors.append(rmse_per_field(cff_predicted[:, ~future], cff_reference[:, ~future]).flatten())
    labels.append(split)
    if future.any():
        errors.append(rmse_per_field(cff_predicted[:, future], cff_reference[:, future]).flatten())
        labels.append(split + "\nfuture times")

plt.figure()
plt.boxplot(errors, showfliers=False)
plt.xticks(range(1, len(labels) + 1), labels)
plt.axhline(TOLERANCE, linestyle="--", color="gray", label="tolerance")
plt.ylabel("RMSE of DCFF [MPa]")
plt.legend()
plt.savefig("results/rmse_boxplot.pdf", bbox_inches="tight")
