"""Creates Voronoi Diagrams."""
import logging
import random
import sys

import numpy as np
import yaml
from PIL import Image
from settings import NamedColours, VoronoiDiagramSettings

from voronoi_generator.colouring_solver import colour_graph
from voronoi_generator.delaunay import generate_partitions as delaunay
from voronoi_generator.general import create_image_array, generate_icons, generate_outlines, generate_points
from voronoi_generator.voronoi import generate_partitions, get_adjacency_matrix

logger = logging.getLogger(__name__)


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

    centroids = generate_points(
        settings.n_centroids,
        x_y_ratio,
        method=settings.sampling_method,
        placed_points=settings.placed_points,
        point_radius=settings.point_radius,
        poisson_radius=settings.poisson_radius,
    )

    partitions = generate_partitions(
        grid, centroids, wrap_x=settings.wrap_x, wrap_y=settings.wrap_y, metric=settings.distance_function
    )

    partitions = partitions.reshape(y_size, x_size).T

    outlines = generate_outlines(partitions, settings.border_thickness) if settings.border_thickness > 0 else None

    if settings.centroid_thickness > 0:
        centroid_mask = generate_icons(centroids, grid, settings.centroid_thickness)
        centroid_mask = centroid_mask.reshape(y_size, x_size).T
    else:
        centroid_mask = None

    delaunay_wrap_x = settings.delaunay_wrap_x if settings.delaunay_wrap_x is not None else settings.wrap_x
    delaunay_wrap_y = settings.delaunay_wrap_y if settings.delaunay_wrap_y is not None else settings.wrap_y
    delaunay_partitions = delaunay(grid, centroids, delaunay_wrap_x, delaunay_wrap_y)
    delaunay_partitions = delaunay_partitions.reshape(y_size, x_size).T
    if settings.delaunay_outline > 0:
        delaunay_outline = (
            generate_outlines(delaunay_partitions, settings.delaunay_outline) if settings.delaunay_outline > 0 else None
        )
    else:
        delaunay_outline = None

    if settings.which_tile == "voronoi":
        adj_matrix = get_adjacency_matrix(partitions)
    else:
        adj_matrix = get_adjacency_matrix(delaunay_partitions)

    graph_colouring = colour_graph(adj_matrix, method=settings.colouring_method)

    partitions_to_colour = partitions if settings.which_tile == "voronoi" else delaunay_partitions
    rgb_array = create_image_array(
        partitions_to_colour,
        graph_colouring,
        settings.colour_list,
        outlines,
        centroid_mask,
        which_colours=settings.which_tile,
        delaunay_outlines=delaunay_outline,
    )

    if settings.file_path is not None:
        logger.info(f"Saving image as {settings.file_path}")
        image = Image.fromarray(rgb_array)
        image.save(settings.file_path, resolution=300)

    return rgb_array


if __name__ == "__main__":
    main()
