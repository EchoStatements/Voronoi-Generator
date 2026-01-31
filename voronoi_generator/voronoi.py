"""Functions used for voronoi generation."""
import logging

import numpy as np
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def generate_partitions(grid, centroids, wrap_x=False, wrap_y=False, metric="euclidean"):
    """Generates partitions/tilings in Voronoi diagram.

    Args:
        grid (np.array): Array of size (n_points, 2) giving coordinates of all points on the grid
        centroids (np.array): Array of size (n_centroids, 2) giving coordinates of all centroids
        wrap_x (bool): Whether to wrap the partitions on the x-axis
        wrap_y (bool): Whether to wrap the partitions on the x-axis
        metric (str): Which metric to use when computing distances

    """
    logger.info("Generating partitions")
    n_centroids = centroids.shape[0]

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

    # Numerical instability in Manhattan distance for floats means that we need
    # to convert to int to avoid artefacts in the finished image
    # Choosing grid.shape[0] as a scaling factor before conversion ensures
    # no loss of resolution
    if metric in ["cityblock", "manhattan"]:
        grid = grid * grid.shape[0]
        grid = grid.astype(int)
        centroids = centroids * grid.shape[0]
        centroids = centroids.astype(int)

    distances = pairwise_distances(grid, centroids, metric=metric)
    partitions = np.argmin(distances, axis=1)
    partitions = np.mod(partitions, n_centroids)

    return partitions


def get_adjacency_matrix(partitions):
    """Generate an adjacency matrix from an array defining which pixels belong in which partition.

    Args:
        partitions (np.array): A 2d numpy array defining which pixels belong to which partition

    Returns:
        np.array: A 2d array of size (n_partitions, n_partitions) representing the adjacency matrix.

    """
    logger.info("Generating adjacency matrix")
    n_partitions = np.max(partitions).astype(int) + 1
    rolled_grids = []
    adjacency_matrix = np.zeros((n_partitions, n_partitions))

    # Create four np.arrays which are copies of partitions, each offset from the
    # original array by one pixel in each of the cardinal directions
    for axis in [0, 1]:
        for roll_dist in [-1, 1]:
            rolled_grids.append(np.roll(partitions, roll_dist, axis))

    # For each partition, we create a mask, then check for what values we can see
    # in the rolled matrices after we apply the mask.
    # If we can see a particular partition index, we know that there is a point in
    # our partition that it's one pixel away from and therefore they are neighbours
    for idx in range(n_partitions):
        sets = []
        for grid in rolled_grids:
            mask = grid == idx
            # add one because partition index can be zero
            overlaps = mask * (partitions + 1)
            neighbours = set(overlaps.reshape(-1) - 1)
            neighbours.remove(-1)
            if idx in neighbours:
                neighbours.remove(idx)
            sets.append(neighbours)
        all_neighbours = set.union(*sets)

        for idx2 in range(n_partitions):
            if idx2 in all_neighbours:
                adjacency_matrix[idx, idx2] = 1

    return adjacency_matrix
