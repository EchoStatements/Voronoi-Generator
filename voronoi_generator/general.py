"""Functions used for both voronoi and delaunay generation."""
import logging

import numpy as np
from scipy.signal import convolve2d
from scipy.stats._qmc import PoissonDisk
from skimage.draw import disk
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def generate_outlines(partitions, border_thickness):
    """Creates a mask defining the partition outlines. Mask is one if pixel is part of outline and zero otherwise.

    Args:
        partitions (np.array): The 2-d array defining which pixels belong to which partitions
        border_thickness (int): How thick the generated outlines should be.

    """
    logger.info("Generating outlines")
    x_offset = np.concatenate((partitions[1:, :], partitions[0:1, :]), axis=0)
    y_offset = np.concatenate((partitions[:, 1:], partitions[:, 0:1]), axis=1)

    # In order to generate outlines, we find pixels where a neighbour in the x direction or y direction
    # is different, giving a one pixel wide outline.
    outlines_1 = np.abs(partitions - x_offset)
    outlines_2 = np.abs(partitions - y_offset)
    outlines = ((outlines_1 + outlines_2) > 0) * 1

    # We thicken the outline by convolving with a matrix containing a circle of ones whose size is determined by
    # the border thickness parameter.
    thickener = np.ones((border_thickness, border_thickness))
    mask = np.zeros((3 * border_thickness, 3 * border_thickness), dtype=np.uint8)

    grid_centre = mask.shape[0] // 2

    rr, cc = disk((grid_centre, grid_centre), border_thickness)
    mask[rr, cc] = 1
    outlines = (convolve2d(outlines, thickener, mode="same", boundary="wrap")) > 0
    return outlines


def create_image_array(
    partitions,
    graph_colouring,
    colour_list,
    voronoi_outlines=None,
    centroid_mask=None,
    which_colours="voronoi",
    delaunay_outlines=None,
):
    """Creates a numpy array with RGB direction which can be converted into an image.

    Note that this involves transposing the x and y axes as a final step.

    Args:
        partitions (np.array): 2d array saying which partition each pixel belongs to.
        graph_colouring (list[int]): List where the ith entry tells us the colouring of the ith partition.
        colour_list (list[int]): List of colours to be used in colouring the graph
        voronoi_outlines (np.array): The outlines to be applied to the image
        mask (np.array): A mask for regions where the second palette in colour_lists should be used
        centroid_mask: a 2d array showing where the centroids are
        which_colours: whether to colour the voronoi or delaunay partitions
        delaunay_outlines: the thickness to be used for the delaunay outlines

    Returns:
        np.array: an array of size (y_size, x_size, 3), containing the RGB data for the image.
    """
    logger.info("Generating image array")
    x_size = partitions.shape[0]
    y_size = partitions.shape[1]

    colour_list = np.array(colour_list)

    rgb_array = np.zeros((x_size, y_size, 3), dtype=np.uint8)

    flat_partitions = partitions.reshape(-1)

    flat_partitions_colour_idx = np.take(graph_colouring, flat_partitions)

    for idx in range(3):
        rgb_array[:, :, idx] = np.take(colour_list[:, idx], flat_partitions_colour_idx).reshape(x_size, y_size)

    if voronoi_outlines is not None:
        for idx in range(3):
            rgb_array[:, :, idx] = (voronoi_outlines * 0) + np.logical_not(voronoi_outlines.astype(bool)) * rgb_array[
                :, :, idx
            ]

    if delaunay_outlines is not None:
        for idx in range(3):
            rgb_array[:, :, idx] = (delaunay_outlines * 0) + np.logical_not(delaunay_outlines.astype(bool)) * rgb_array[
                :, :, idx
            ]

    if centroid_mask is not None:
        for idx in range(3):
            rgb_array[:, :, idx] = (centroid_mask * 0) + np.logical_not(centroid_mask.astype(bool)) * rgb_array[
                :, :, idx
            ]

    return np.swapaxes(rgb_array, 0, 1)


def generate_points(
    n_centroids, x_y_ratio, method="uniform", placed_points=None, point_radius=None, poisson_radius=None
):
    """Generate points for Voronoi diagram.

    Args:
        n_centroids (int): Number of points to be generated
        x_y_ratio: The ratio between x axis scale and y axis scale
        method: The method used to generate the points
        placed_points: A list of the centroids to be explicitly placed
        point_radius: The minimum distance of random points from placed points
        poisson_radius: The minimum distance for Poisson disk sampling

    """
    logger.info("Generating points")
    if method == "uniform":
        centroids = np.zeros((n_centroids, 2))
        centroids[:, 0] = np.random.uniform(0, 1, size=n_centroids)
        centroids[:, 1] = np.random.uniform(0, x_y_ratio, size=n_centroids)
    elif method == "poisson":
        logger.info("Poisson sampling does not respect `n_centroids` value")
        poisson_seed = np.random.choice(100000)
        engine = PoissonDisk(d=2, radius=poisson_radius, seed=poisson_seed)
        centroids = engine.random(n_centroids * 10)
        centroids[:, 1] = centroids[:, 1] * x_y_ratio
    else:
        raise ValueError

    if placed_points is not None:
        if n_centroids == 0:
            centroids = np.array(placed_points)
        else:
            dist_from_centroid = pairwise_distances(centroids, placed_points)
            min_dist_from_points = np.min(dist_from_centroid, axis=1)
            allowed_point_indices = np.argwhere(min_dist_from_points > point_radius)
            allowed_points = [centroids[idx, :].flatten() for idx in allowed_point_indices]
            allowed_points = allowed_points + placed_points
            centroids = np.array(allowed_points)

    return centroids


def generate_icons(centroids, grid, icon_thickness):
    """Generate a mask for showing the centres of each partition.

    Args:
        centroids (np.ndarray): An array of coordinates of centroids
        grid (np.ndarray): An array of coordinated for points in the grid
        icon_thickness (int): How thick to make the icon for each partition centre
    """
    icon_thickness = (grid[1, 0] - grid[0, 0]) * icon_thickness
    distances = pairwise_distances(grid, centroids)
    centroid_mask = np.min(distances, axis=1) < icon_thickness
    return centroid_mask
