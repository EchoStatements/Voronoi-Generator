"""Graph colouring solver."""
import numpy as np
import pulp


def colour_graph(adjacency_matrix):
    """Computes graph colouring for the adjacency matrix.

    Args:
        adjacency_matrix (np.array): The (n_vertices, n_vertices) adjacency matrix for the graph to be coloured

    Returns:
        list[int]: The colouring of the graph given as a list of integers of size (n_vertices). All vertices
                   with the same integer may be coloured the same without having two adjacent vertices share a colour.
    """
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
