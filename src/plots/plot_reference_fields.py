"""Overview of the data: reference CFF of some training scenarios at some times.

The fields are drawn as a cascade of panels, all with the same colour scale.
"""

import matplotlib.pyplot as plt

from src.data.dataset import FIRST_YEAR, load_split
from src.plots.fault import draw_on_fault

SCENARIOS = [11, 21]
TIME_INDICES = [0, 3, 6, 8]

PANEL_SIZE = 0.4
SHIFT_PER_TIME = [0.11, -0.04]
SHIFT_PER_SCENARIO = [0.07, -0.24]

_, times, cff = load_split("train")
shown_cff = cff[SCENARIOS][:, TIME_INDICES]
colors = {
    "cmap": plt.get_cmap("viridis", 10),
    "vmin": shown_cff.min(),
    "vmax": shown_cff.max(),
}

figure = plt.figure(figsize=(9, 5))
for row, scenario in enumerate(SCENARIOS):
    for column, time_index in enumerate(TIME_INDICES):
        left = 0.02 + column * SHIFT_PER_TIME[0] + row * SHIFT_PER_SCENARIO[0]
        bottom = 0.52 + column * SHIFT_PER_TIME[1] + row * SHIFT_PER_SCENARIO[1]
        axes = figure.add_axes([left, bottom, PANEL_SIZE, PANEL_SIZE])
        image = draw_on_fault(axes, cff[scenario, time_index], **colors)
        for side in axes.spines.values():
            side.set_color("white")
            side.set_linewidth(3)

        label = f"{FIRST_YEAR + times[time_index]:g}"
        if column == 0:
            label = f"scenario {scenario}\n" + label
        axes.text(
            0.03,
            0.95,
            label,
            transform=axes.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "boxstyle": "round"},
        )

colorbar_axes = figure.add_axes([0.9, 0.15, 0.02, 0.6])
figure.colorbar(image, cax=colorbar_axes, label="CFF [MPa]")
figure.savefig("results/reference_fields.pdf", bbox_inches="tight")
