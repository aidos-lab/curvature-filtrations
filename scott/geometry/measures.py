"""Curvature methods for graphs."""

import ot
import warnings

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


#  ╭──────────────────────────────────────────────────────────╮
#  │ Ollivier-Ricci                                           │
#  ╰──────────────────────────────────────────────────────────╯


def ollivier_ricci_curvature(
    G, alpha=0.0, weight=None, prob_fn=None
) -> np.ndarray:
    """Calculate Ollivier--Ricci curvature for graphs that allows for a custom probability measure.

    This function calculates the Ollivier--Ricci curvature of a graph,
    optionally taking (positive) edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    alpha : float
        Provides laziness parameter for default probability measure. The
        measure is not compatible with a user-defined `prob_fn`. If such
        a function is set, `alpha` will be ignored.

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated. Notice that
        if `prob_fn` is provided, this parameter will have no effect for
        the calculation of probability measures, but it will be used for
        the calculation of shortest-path distances.

    prob_fn : callable or None
        If set, should refer to a function that calculate a probability
        measure for a given graph and a given node. This callable needs
        to satisfy the following signature:

        ``prob_fn(G, node, node_to_index)``

        Here, `G` refers to the graph, `node` to the node whose measure
        is to be calculated, and `node_to_index` to the lookup map that
        maps a node identifier to a zero-based index.

        If `prob_fn` is set, providing `alpha` will not have an effect.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    assert 0 <= alpha <= 1

    # Ensures that we can map a node to its index in the graph,
    # regardless of whether node labels or node names are being
    # used.
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    # This is defined inline anyway, so there is no need to follow the
    # same conventions as for the `prob_fn` parameter.
    def _make_probability_measure(node):
        values = np.zeros(len(G.nodes))
        values[node_to_index[node]] = alpha

        degree = G.degree(node, weight=weight)

        for neighbor in G.neighbors(node):

            if weight is not None:
                w = G[node][neighbor][weight]
            else:
                w = 1.0

            values[node_to_index[neighbor]] = (1 - alpha) * w / degree

        return values

    # We pre-calculate all information about the probability measure,
    # making edge calculations easier later on.
    if prob_fn is None:
        measures = list(map(_make_probability_measure, G.nodes))
    else:
        measures = list(map(lambda x: prob_fn(G, x, node_to_index), G.nodes))

    # This is the cost matrix for calculating the Ollivier--Ricci
    # curvature in practice.
    M = nx.floyd_warshall_numpy(G, weight=weight)

    curvature = []
    for edge in G.edges():
        source, target = edge

        mi = measures[node_to_index[source]]
        mj = measures[node_to_index[target]]

        distance = ot.emd2(mi, mj, M)
        curvature.append(1.0 - distance)

    return np.asarray(curvature)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Resistance                                               │
#  ╰──────────────────────────────────────────────────────────╯


def resistance_curvature(G, weight=None):
    """Calculate Resistance Curvature of a graph.

    This function calculates the resistance curvature of a graph,
    optionally taking (positive) edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # Generate Matrix of Resistance Distances and Node Reference
        # Dictionary
        R, node_to_index = _pairwise_resistances(G, weight=weight)
    curvature = []

    for edge in G.edges():
        source, target = edge
        source_curvature = _node_resistance_curvature(
            G, source, weight=weight, R=R, node_to_index=node_to_index
        )
        target_curvature = _node_resistance_curvature(
            G, target, weight=weight, R=R, node_to_index=node_to_index
        )

        edge_curvature = (
            2
            * (source_curvature + target_curvature)
            / R[node_to_index[source], node_to_index[target]]
        )
        curvature.append(edge_curvature)

    return np.asarray(curvature)


def _pairwise_resistances(G, weight=None):
    """Calculate pairwise resistances for all neighbors of a graph.

    Calculate pairwise resistances for all neighbors in a graph `G`
    using the networkx implementation of `resistance_distance`. This
    function helps reducing redundant computations when calculating
    `resistance_curvature`, by doing the necessary calculations up
    front.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    R : np.matrix
        A matrix of pairwise resistance distances between neighboring
        nodes in `G`.

    node_to_index : dict
        A reference dictionary for translating between nodes and indices
        of `G`.
    """
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    n = len(G.nodes())

    # Initialize nxn Matrix
    R = np.zeros(shape=(n, n))

    # List of connected components with original node order
    components = list(
        [G.subgraph(c).copy() for c in nx.connected_components(G)]
    )
    for C in components:
        for source, target in C.edges():
            i, j = node_to_index[source], node_to_index[target]
            r = nx.resistance_distance(
                C,
                source,
                target,
                weight=weight,
                invert_weight=False,
            )
            # Assign Matrix Entries for neighbors
            R[i, j], R[j, i] = r, r

    return R, node_to_index


def _node_resistance_curvature(
    G, node, weight=None, R=None, node_to_index=None
):
    """Calculate Resistance Curvature of a given node in a graph 'G'.

    This function calculates the resistance curvature of only
    the nodes in a graph, optionally takes (positive)
    edge weights into account. This is a helper function for
    resistance_curvature; the curvature of each node is used to
    determine the overall curvature of the graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated.

    Returns
    -------
    np.float32
        The node curvature of a given node in `G`.
    """
    assert node in G.nodes()

    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    if R is None:
        R, node_to_index = _pairwise_resistances(G, weight=weight)

    neighbors = list(G.neighbors(node))
    rel_resistance = 0

    for neighbor in neighbors:

        if weight is not None and len(G.get_edge_data(node, neighbor)) > 0:
            w = G[node][neighbor][weight]

        else:
            w, G[node][neighbor]["weight"] = 1, 1

        rel_resistance += R[node_to_index[node]][node_to_index[neighbor]] * w

    node_curvature = 1 - 0.5 * rel_resistance

    return np.float32(node_curvature)
