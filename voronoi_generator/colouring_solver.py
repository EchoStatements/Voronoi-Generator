"""Graph colouring solver."""
import logging

import networkx as nx
import numpy as np
import pulp

logger = logging.getLogger(__name__)


def colour_graph(adjacency_matrix, method="connected_sequential_dfs"):
    """Computes graph colouring for the adjacency matrix.

    Args:
        adjacency_matrix: The adjacency matrix of the graph to colour
        method: which method to use for colouring the graph. "integer_programming" uses pulp, while
            other methods use networkx's greedy colouring algorithms.
    """
    if method == "integer_programming":
        return integer_programming_colouring(adjacency_matrix)
    else:
        return greedy_colouring(adjacency_matrix, strategy="DSATUR")


def greedy_colouring(adj, strategy="DSATUR"):
    """Computes a greedy coloring of a graph given its adjacency matrix.

    Args:
        adj: A 2D array or matrix representing the adjacency structure of the graph.
            The value at (i, j) indicates whether there is an edge between vertex `i`
            and vertex `j`.
        strategy: A string specifying the strategy for greedy coloring. The valid
            strategies can be found in NetworkX's documentation for the `greedy_color`
            method. Defaults to "DSATUR".

    Returns:
        A list of integers, where the position in the list corresponds to a vertex in
        the graph, and the value at that position represents the color assigned
        to that vertex.

    Raises:
        RuntimeError: If the algorithm determines that more than four colors are
            required to color the graph.
    """
    n_partitions = adj.shape[0]
    graph = nx.Graph()
    for idx_1 in range(n_partitions):
        for idx2 in range(idx_1 + 1, n_partitions):
            if adj[idx_1, idx2]:
                graph.add_edge(idx_1, idx2)

    colouring = nx.coloring.greedy_color(graph, strategy=strategy)

    num_colours = max(colouring.values()) + 1
    if num_colours > 4:  # noqa: PLR2004
        raise RuntimeError("Could not find 4 colouring of the graph.")

    # converting from dict to list
    colour_list = [colouring[i] for i in range(n_partitions)]
    return colour_list


def integer_programming_colouring(adjacency_matrix):
    """Computes graph colouring for the adjacency matrix.

    Args:
        adjacency_matrix (np.array): The (n_vertices, n_vertices) adjacency matrix for the graph to be coloured

    Returns:
        list[int]: The colouring of the graph given as a list of integers of size (n_vertices). All vertices
                   with the same integer may be coloured the same without having two adjacent vertices share a colour.
    """
    logger.info("Computing graph colouring")
    n_vertices = adjacency_matrix.shape[0]
    edges = np.argwhere(adjacency_matrix == 1)

    model = pulp.LpProblem(sense=pulp.LpMinimize)

    variables = [
        [pulp.LpVariable(name=f"edgge_{i}_{j}", cat=pulp.LpBinary) for j in range(n_vertices)]
        for i in range(n_vertices)
    ]

    chromatic_number = pulp.LpVariable(name="chromatic number", cat="Integer")

    for vertex_idx in range(n_vertices):
        model += pulp.lpSum(variables[vertex_idx]) == 1

    for v_1, v_2 in edges:
        for k in range(n_vertices):
            model += variables[v_1][k] + variables[v_2][k] <= 1

    # we also restrict the chromatic number to be the number of the highest used colour
    for u in range(n_vertices):
        for k in range(n_vertices):
            model += chromatic_number >= (k + 1) * variables[u][k]

    # objective function - minimise the chromatic number
    model += chromatic_number

    _ = model.solve(pulp.PULP_CBC_CMD(msg=False))

    colours = []
    for u in range(n_vertices):
        for k in range(n_vertices):
            if variables[u][k].value() != 0:
                colours.append(k)

    return colours
