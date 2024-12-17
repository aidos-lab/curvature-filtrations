import ot

import networkx as nx
import numpy as np
import scipy as sp


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
#  │ Probability Measures                                     │
#  ╰──────────────────────────────────────────────────────────╯


def prob_rw(G, node, node_to_index):
    """Probability measure based on random walk probabilities."""

    A = nx.to_scipy_sparse_array(G, format="csr").todense()
    n, m = A.shape
    D = sp.sparse.csr_array(
        sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format="csr")
    ).todense()

    P = np.linalg.inv(D) @ A

    values = np.zeros(len(G.nodes))
    values[node_to_index[node]] = 1.0

    x = values
    values = x + P @ x + P @ P @ x

    values /= values.sum()
    return values


def prob_two_hop(G, node, node_to_index):
    """Probability measure based on two-hop neighbourhoods."""
    alpha = 0.5
    values = np.zeros(len(G.nodes))
    values[node_to_index[node]] = alpha

    subgraph = nx.ego_graph(G, node, radius=2)

    w = 0.25

    direct_neighbors = list(G.neighbors(node))
    for neighbor in direct_neighbors:
        values[node_to_index[neighbor]] = (1 - alpha) * w

    w = 0.05

    for neighbor in subgraph.nodes():
        if neighbor not in direct_neighbors and neighbor != node:
            index = node_to_index[neighbor]
            values[index] = (1 - alpha) * w

    values /= values.sum()
    return values
