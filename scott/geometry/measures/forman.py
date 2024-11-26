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
        curvature.append(
            float(4 - source_degree - target_degree + 3 * n_triangles)
        )

    return np.asarray(curvature)


def _forman_curvature_weighted(G, weight):
    has_node_attributes = bool(nx.get_node_attributes(G, weight))

    curvature = []
    for edge in G.edges:
        source, target = edge
        source_weight, target_weight = 1.0, 1.0

        # Makes checking for duplicate edges easier below. We expect the
        # source vertex to be the (lexicographically) smaller one.
        if source > target:
            source, target = target, source

        if has_node_attributes:
            source_weight = G.nodes[source][weight]
            target_weight = G.nodes[target][weight]

        edge_weight = G[source][target][weight]

        e_curvature = source_weight / edge_weight
        e_curvature += target_weight / edge_weight

        parallel_edges = list(G.edges(source, data=weight)) + list(
            G.edges(target, data=weight)
        )

        for u, v, w in parallel_edges:
            if u > v:
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
    The balanced Forman curvature is a measure of the "sharpness" or "bottleneck" properties of edges in a graph, which can be useful for understanding over-squashing and bottlenecks in graph neural networks.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (weighted or unweighted).
    Returns
    -------
    list of tuple
        A list of tuples (u, v, curvature) for each edge (u, v) in the graph,
        where `u` and `v` are nodes and `curvature` is the computed balanced
        Forman curvature for the edge.
    References
    ----------
    .. [1] Topping, Jake, et al. "Understanding Over-Squashing and Bottlenecks
           on Graphs via Curvature." International Conference on Learning
           Representations. 2022.
    """
    # Weighted adjacency matrix
    A = nx.to_numpy_array(G, weight=weight)
    N = A.shape[0]

    # Weighted degree information
    d_in = A.sum(axis=0)  # Weighted in-degrees
    d_out = A.sum(axis=1)  # Weighted out-degrees

    # Second power of the weighted adjacency matrix
    A2 = np.matmul(A, A)

    curvature_values = []

    for u, v in G.edges():
        i, j = u, v  # Node indices in the adjacency matrix
        weight = A[i, j]

        if weight == 0:  # Skip if there's no edge (just a safeguard)
            continue

        # Determine max and min degrees
        d_max = max(d_in[i], d_out[j])
        d_min = min(d_in[i], d_out[j])

        if d_max * d_min == 0:
            curvature = 0
        else:
            # Compute sharpness and lambda
            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP_1 = A[k, j] * (A2[i, k] - A[i, k]) * weight
                TMP_2 = A[i, k] * (A2[k, j] - A[k, j]) * weight

                if TMP_1 > 0:
                    sharp_ij += 1
                    lambda_ij = max(lambda_ij, TMP_1)

                if TMP_2 > 0:
                    sharp_ij += 1
                    lambda_ij = max(lambda_ij, TMP_2)

            # Balanced Forman Curvature
            curvature = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * weight
            )
            if lambda_ij > 0:
                curvature += sharp_ij / (d_max * lambda_ij)

        # Store curvature for this edge
        curvature_values.append(curvature)

    return np.asarray(curvature_values)
