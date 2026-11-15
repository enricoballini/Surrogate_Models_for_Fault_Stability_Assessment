"""Draw a field on the fault."""

import numpy as np

from src.data.dataset import load_fault_mesh


def fault_plane_coordinates(nodes):
    """Coordinates of the nodes along the length and the height of the fault plane.

    Each coordinate is scaled to [0, 1]. The fault is much longer than high: scaling
    the two directions separately stretches it, so that all the cells are visible.
    """
    centred_nodes = nodes - nodes.mean(axis=0)
    # the first two principal directions of the nodes lie on the fault plane
    _, _, directions = np.linalg.svd(centred_nodes)
    length_direction = directions[0] * np.sign(directions[0][0])
    height_direction = directions[1] * np.sign(directions[1][2])
    coordinates = []
    for direction in [length_direction, height_direction]:
        coordinate = centred_nodes @ direction
        coordinate = coordinate - coordinate.min()
        coordinates.append(coordinate / coordinate.max())
    return coordinates


def draw_on_fault(axes, field, **color_options):
    """Draw one value per fault cell and return the image, needed to add a colorbar."""
    nodes, triangles = load_fault_mesh()
    along_length, along_height = fault_plane_coordinates(nodes)
    image = axes.tripcolor(
        along_length,
        along_height,
        triangles,
        facecolors=field,
        edgecolors="black",
        linewidth=0.1,
        **color_options,
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xticks([])
    axes.set_yticks([])
    return image
