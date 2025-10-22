import networkx as nx
import numpy as np


#  ╭──────────────────────────────────────────────────────────╮
#  │ Forman-Ricci                                             │
#  ╰──────────────────────────────────────────────────────────╯


def forman_curvature(G, weight=None):
    """Calculate Forman--Ricci curvature of a graph.

    This function calculates the Forman--Ricci curvature of a graph,
    optionally taking (positive) node and edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. Will use the same attribute to look up node weights. If
        None, unweighted curvature is calculated.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    # This calculation is much more efficient than the weighted one, so
    # we default to it in case there are no weights in the graph.
    if weight is None:
        return _forman_curvature_unweighted(G)
    else:
        return _forman_curvature_weighted(G, weight)


def _forman_curvature_unweighted(G):
    curvature = []
    for edge in G.edges():

        source, target = edge
        source_degree = G.degree(source)
        target_degree = G.degree(target)

        source_neighbours = set(G.neighbors(source))
        target_neighbours = set(G.neighbors(target))

        n_triangles = len(source_neighbours.intersection(target_neighbours))
        curvature.append(float(4 - source_degree - target_degree + 3 * n_triangles))

    return np.asarray(curvature)


def _forman_curvature_weighted(G, weight):
    has_node_attributes = bool(nx.get_node_attributes(G, weight))

    curvature = []
    for edge in G.edges:
        source, target = edge
        source_weight, target_weight = 1.0, 1.0

        # Makes checking for duplicate edges easier below. We expect the
        # source vertex to be the (lexicographically) smaller one.
        # Use string representation for comparison to handle mixed node types
        if str(source) > str(target):
            source, target = target, source

        if has_node_attributes:
            source_weight = G.nodes[source][weight]
            target_weight = G.nodes[target][weight]

        edge_weight = G[source][target][weight]

        e_curvature = source_weight / edge_weight
        e_curvature += target_weight / edge_weight

        parallel_edges = list(G.edges(source, data=weight)) + list(G.edges(target, data=weight))

        for u, v, w in parallel_edges:
            # Use string representation for comparison to handle mixed node types
            if str(u) > str(v):
                u, v = v, u

            if (u, v) == edge:
                continue
            else:
                e_curvature -= w / np.sqrt(edge_weight * w)

        e_curvature *= edge_weight
        curvature.append(float(e_curvature))

    return np.asarray(curvature)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Balanced Forman                                          │
#  ╰──────────────────────────────────────────────────────────╯


def balanced_forman_curvature(G, weight=None):
    """
    Compute the balanced Forman curvature for each edge in a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (weighted or unweighted).
    weight : str or None
        Name of the edge weight attribute. If None, the graph is treated as unweighted.

    Returns
    -------
    np.array
        An array of balanced Forman curvature values, following the ordering of edges of `G`.

    References
    ----------
    Topping, Jake, et al. "Understanding Over-Squashing and Bottlenecks on Graphs via Curvature." International Conference on Learning Representations. 2022.
    """
    # Compute adjacency matrix and degree information
    A, d_in, d_out, node_to_index = _prepare_graph_data(G, weight)

    # Compute curvature values
    curvature_values = [
        _compute_edge_curvature(A, d_in, d_out, node_to_index, u, v) for u, v in G.edges()
    ]

    return np.asarray(curvature_values)


def _prepare_graph_data(G, weight):
    """Prepare weighted adjacency matrix and degree information."""
    A = nx.to_numpy_array(G, weight=weight)
    d_in = A.sum(axis=0)  # Weighted in-degrees
    d_out = A.sum(axis=1)  # Weighted out-degrees

    # Create node to index mapping
    node_to_index = {node: i for i, node in enumerate(G.nodes())}

    return A, d_in, d_out, node_to_index


def _compute_edge_curvature(A, d_in, d_out, node_to_index, u, v):
    """Compute the balanced Forman curvature for a single edge."""
    i, j = node_to_index[u], node_to_index[v]  # Convert node identifiers to matrix indices
    weight = A[i, j]

    if weight == 0:  # Safeguard against missing edges
        return 0

    d_max, d_min = max(d_in[i], d_out[j]), min(d_in[i], d_out[j])

    if d_max * d_min == 0:
        return 0

    sharp_ij, lambda_ij = _compute_sharpness_and_lambda(A, i, j, weight)

    # Balanced Forman Curvature
    curvature = (
        (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * np.matmul(A, A)[i, j] * weight
    )

    if lambda_ij > 0:
        curvature += sharp_ij / (d_max * lambda_ij)

    return curvature


def _compute_sharpness_and_lambda(A, i, j, weight):
    """Compute the sharpness and lambda values for an edge."""
    N = A.shape[0]
    sharp_ij = 0
    lambda_ij = 0

    for k in range(N):
        TMP_1 = A[k, j] * (_second_power(A, i, k) - A[i, k]) * weight
        TMP_2 = A[i, k] * (_second_power(A, k, j) - A[k, j]) * weight

        if TMP_1 > 0:
            sharp_ij += 1
            lambda_ij = max(lambda_ij, TMP_1)

        if TMP_2 > 0:
            sharp_ij += 1
            lambda_ij = max(lambda_ij, TMP_2)

    return sharp_ij, lambda_ij


def _second_power(A, i, k):
    """Compute the second power of the adjacency matrix."""
    return np.matmul(A, A)[i, k]
