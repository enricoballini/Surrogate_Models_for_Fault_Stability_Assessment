"""Plot reference CFF, predicted CFF and their difference DCFF on the fault.

One figure is saved for each chosen scenario, with one column for each time.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import FIRST_YEAR
from src.metrics.metrics import load_references_and_predictions
from src.plots.fault import draw_on_fault

SPLIT = "validation"
SCENARIOS = [0, 1]

times, cff_reference, cff_predicted = load_references_and_predictions()[SPLIT]
dcff = cff_predicted - cff_reference

for scenario in SCENARIOS:
    fields = np.array([cff_reference[scenario], cff_predicted[scenario]])
    cff_colors = {"cmap": "viridis", "vmin": fields.min(), "vmax": fields.max()}
    largest_dcff = np.abs(dcff[scenario]).max()
    dcff_colors = {"cmap": "RdBu_r", "vmin": -largest_dcff, "vmax": largest_dcff}

    figure, axes = plt.subplots(3, len(times), figsize=(2 * len(times), 4.5))
    for column in range(len(times)):
        axes[0, column].set_title(f"{FIRST_YEAR + times[column]:g}")
        draw_on_fault(axes[0, column], cff_reference[scenario, column], **cff_colors)
        cff_image = draw_on_fault(axes[1, column], cff_predicted[scenario, column], **cff_colors)
        dcff_image = draw_on_fault(axes[2, column], dcff[scenario, column], **dcff_colors)
    axes[0, 0].set_ylabel("reference")
    axes[1, 0].set_ylabel("prediction")
    axes[2, 0].set_ylabel("difference")

    figure.colorbar(cff_image, ax=axes[:2], label="CFF [MPa]")
    figure.colorbar(dcff_image, ax=axes[2], label="DCFF [MPa]")
    figure.savefig(f"results/cff_on_fault_{SPLIT}_scenario_{scenario}.pdf", bbox_inches="tight")
    plt.close(figure)
