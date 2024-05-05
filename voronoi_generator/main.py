"""Creates Voronoi Diagrams."""
import logging
import random
import sys

import numpy as np
import yaml
from colouring_solver import colour_graph
from PIL import Image
from scipy.signal import convolve2d
from scipy.stats.qmc import PoissonDisk
from settings import NamedColours, VoronoiDiagramSettings
from skimage.draw import disk
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def generate_partitions(grid, centroids, wrap_x=False, metric="euclidean"):
    """Generates partitions/tilings in Voronoi diagram.

    Args:
        grid (np.array): Array of size (n_points, 2) giving coordinates of all points on the grid
        centroids (np.array): Array of size (n_centroids, 2) giving coordinates of all centroids
        wrap_x (bool): Whether to wrap the partitions on the x-axis
        metric (str): Which metric to use when computing distances

    """
    logger.info("Generating partitions")
    n_centroids = centroids.shape[0]

    if wrap_x:
        grid_width = np.max(grid[:, 0]) - np.min(grid[:, 0])
        grid_width_adjust = np.zeros_like(centroids)
        grid_width_adjust[:, 0] = grid_width
        centroids = np.concatenate(
            (centroids, centroids - grid_width_adjust, centroids + grid_width_adjust),
            axis=0,
        )

    distances = pairwise_distances(grid, centroids, metric=metric)
    partitions = np.argmin(distances, axis=1)
    partitions = np.mod(partitions, n_centroids)

    return partitions


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
        for roll_dist in [1, 2]:
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
            neighbours.remove(idx)
            sets.append(neighbours)
        all_neighbours = set.union(*sets)

        for idx2 in range(n_partitions):
            if idx2 in all_neighbours:
                adjacency_matrix[idx, idx2] = 1

    return adjacency_matrix


def create_image_array(partitions, graph_colouring, colour_list, outlines=None):
    """Creates a numpy array with RGB direction which can be converted into an image.

    Note that this involves transposing the x and y axes as a final step.

    Args:
        partitions (np.array): 2d array saying which partition each pixel belongs to.
        graph_colouring (list[int]): List where the ith entry tells us the colouring of the ith partition.
        colour_list (list[int]): List of colours to be used in colouring the graph
        outlines (np.array): The outlines to be applied to the image
        mask (np.array): A mask for regions where the second palette in colour_lists should be used

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

    if outlines is not None:
        for idx in range(3):
            rgb_array[:, :, idx] = (outlines * 0) + np.logical_not(outlines.astype(bool)) * rgb_array[:, :, idx]

    return np.swapaxes(rgb_array, 0, 1)


def generate_points(settings, x_y_ratio, method="uniform", placed_points=[[0.25, 0.35], [0.75, 0.35]], radius=0.22):
    """Generate points for Voronoi diagram.

    Args:
        settings: The settings to be used to generate points
        x_y_ratio: The ratio between x axis scale and y axis scale
        method: The method used to generate the points
        placed_points: The points to be placed explicitly
        radius: How far generated points must be at a minimum from the placed points

    """
    logger.info("Generating points")
    if method == "uniform":
        centroids = np.zeros((settings.n_centroids, 2))
        centroids[:, 0] = np.random.uniform(0, 1, size=settings.n_centroids)
        centroids[:, 1] = np.random.uniform(0, x_y_ratio, size=settings.n_centroids)
    elif method == "poisson":
        logger.info("Poisson sampling does not respect `n_centroids` value")
        poisson_seed = np.random.choice(100000)
        engine = PoissonDisk(d=2, radius=0.11, seed=poisson_seed)
        centroids = engine.random(settings.n_centroids * 10)
        centroids[:, 1] = centroids[:, 1] * x_y_ratio
    else:
        raise ValueError

    if placed_points is not None:
        dist_from_centroid = pairwise_distances(centroids, placed_points)
        min_dist_from_points = np.min(dist_from_centroid, axis=1)
        allowed_point_indices = np.argwhere(min_dist_from_points > radius)
        allowed_points = [centroids[idx, :].flatten() for idx in allowed_point_indices]
        allowed_points = allowed_points + placed_points
        centroids = np.array(allowed_points)

    return centroids


def create_voronoi_diagram(settings: VoronoiDiagramSettings):
    """Creates and saves Voronoi diagram image.

    Args:
        settings (VoronoiDiagramSettings): settings to be used for generating image

    """
    x_size = settings.x_size
    y_size = settings.y_size
    x_y_ratio = settings.y_size / settings.x_size

    random.seed(settings.python_seed)
    np.random.seed(settings.numpy_seed)

    x_coords = np.linspace(0, 1, x_size)
    y_coords = np.linspace(0, x_y_ratio, y_size)
    grid = np.array(np.meshgrid(x_coords, y_coords))
    grid = grid.reshape(2, -1).T

    centroids = generate_points(settings, x_y_ratio, method=settings.sampling_method)

    partitions = generate_partitions(grid, centroids, wrap_x=True, metric=settings.distance_function)

    partitions = partitions.reshape(y_size, x_size).T

    outlines = generate_outlines(partitions, settings.border_thickness) if settings.border_thickness > 0 else None

    adj_matrix = get_adjacency_matrix(partitions)

    graph_colouring = colour_graph(adj_matrix)

    rgb_array = create_image_array(partitions, graph_colouring, settings.colour_list, outlines)

    if settings.file_path is not None:
        logger.info(f"Saving image as {settings.file_path}")
        image = Image.fromarray(rgb_array)
        image.save(settings.file_path, resolution=300)

    return rgb_array


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    file_path = sys.argv[1]
    with open(file_path) as file_:
        settings = VoronoiDiagramSettings(**yaml.safe_load(file_))
    if settings.named_colours_file is not None:
        with open(settings.named_colours_file) as file_:
            named_colours = NamedColours(**yaml.safe_load(file_))
    for idx in range(len(settings.colour_list)):
        if isinstance(settings.colour_list[idx], str):
            settings.colour_list[idx] = named_colours.named_colours[settings.colour_list[idx]]

    create_voronoi_diagram(settings)


if __name__ == "__main__":
    main()
