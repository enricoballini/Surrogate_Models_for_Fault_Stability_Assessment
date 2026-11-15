"""Plot the RMSE of DCFF in time: median over the scenarios, with the band from P10 to P90."""

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import FIRST_YEAR, load_split
from src.metrics.metrics import TOLERANCE, load_references_and_predictions, rmse_per_field

plt.figure()

for split, evaluation in load_references_and_predictions().items():
    times, cff_reference, cff_predicted = evaluation
    errors = rmse_per_field(cff_predicted, cff_reference)
    years = FIRST_YEAR + times
    plt.plot(years, np.median(errors, axis=0), marker="o", label=split)
    plt.fill_between(
        years,
        np.percentile(errors, 10, axis=0),
        np.percentile(errors, 90, axis=0),
        alpha=0.2,
    )

_, training_times, _ = load_split("train")
last_training_year = FIRST_YEAR + training_times.max()
plt.axvline(last_training_year, linestyle=":", color="black", label="last training time")
plt.axhline(TOLERANCE, linestyle="--", color="gray", label="tolerance")
plt.xlabel("year")
plt.ylabel("RMSE of DCFF [MPa]")
plt.grid(linestyle="--", alpha=0.5)
plt.legend()
plt.savefig("results/rmse_in_time.pdf", bbox_inches="tight")
