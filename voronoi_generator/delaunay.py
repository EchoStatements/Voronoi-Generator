"""Functions for delaunay triangulation."""
import logging

import numpy as np
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


def generate_partitions(grid, centroids, wrap_x=False, wrap_y=False, metric="euclidean"):  # noqa: C901, PLR0915
    """Generates partitions/tilings in Voronoi diagram.

    Args:
        grid (np.array): Array of size (n_points, 2) giving coordinates of all points on the grid
        centroids (np.array): Array of size (n_centroids, 2) giving coordinates of all centroids
        wrap_x (bool): Whether to wrap the partitions on the x-axis
        wrap_y (bool): Whether to wrap the partitions on the x-axis
        metric (str): Which metric to use when computing distances

    """
    logger.info("Generating partitions")

    if wrap_x:
        grid_width = np.max(grid[:, 0]) - np.min(grid[:, 0])
        grid_width_adjust = np.array([grid_width, 0]).reshape(1, 2)
        centroids = np.concatenate(
            (centroids, centroids - grid_width_adjust, centroids + grid_width_adjust),
            axis=0,
        )
    if wrap_y:
        grid_height = np.max(grid[:, 1]) - np.min(grid[:, 1])
        grid_height_adjust = np.array([0, grid_height]).reshape(1, 2)
        centroids = np.concatenate(
            (centroids, centroids - grid_height_adjust, centroids + grid_height_adjust),
            axis=0,
        )
    if wrap_x and wrap_y:
        centroids = np.concatenate(
            (
                centroids,
                centroids - grid_height_adjust - grid_width_adjust,
                centroids - grid_height_adjust + grid_width_adjust,
                centroids + grid_height_adjust + grid_width_adjust,
                centroids + grid_height_adjust - grid_width_adjust,
            ),
            axis=0,
        )

    triangulation = Delaunay(centroids)

    partitions = triangulation.find_simplex(grid)

    if wrap_x:
        left_x = np.min(grid[:, 0])
        right_x = np.max(grid[:, 0])

        left_mask = grid[:, 0] == left_x
        right_mask = grid[:, 0] == right_x

        left_y = grid[left_mask, 1]
        left_parts = partitions[left_mask]

        right_y = grid[right_mask, 1]
        right_parts = partitions[right_mask]

        right_y_to_part = dict(zip(right_y, right_parts))

        # For each unique partition on left, find its most common neighbour on right
        # Then replace each of those values with the value of its most common neighbour
        for lp in np.unique(left_parts):
            mask = left_parts == lp
            y_coords = left_y[mask]
            right_partners = np.array([right_y_to_part[y] for y in y_coords if y in right_y_to_part])

            if len(right_partners) > 0:
                values, counts = np.unique(right_partners, return_counts=True)
                most_common_right = values[np.argmax(counts)]
                if lp != most_common_right:
                    partitions[partitions == lp] = most_common_right

    _, partitions = np.unique(partitions, return_inverse=True)

    return partitions
