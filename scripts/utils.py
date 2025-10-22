"""Various utility functions and classes."""

import collections

import gudhi as gd
import networkx as nx
import numpy as np


def calculate_persistent_homology(G, k=3):
    """
    Calculate the persistent homology of a graph's clique complex up to dimension k.

    Parameters
    ----------
    G : networkx.Graph
        The input graph where nodes and edges may have a 'curvature' attribute.
    k : int, optional
        The maximum dimension of the homology to compute (default is 3).

    Returns
    -------
    diagrams : list of list of tuple
        A list of persistence diagrams for each dimension from 0 to k. Each
        persistence diagram is a list of tuples (birth, death) representing
        the birth and death times of homological features.
    """
    """Calculate persistent homology of graph clique complex."""
    st = gd.SimplexTree()

    for v, w in G.nodes(data=True):
        if "curvature" in w:
            weight = w["curvature"]
            st.insert([v], filtration=weight)

    for u, v, w in G.edges(data=True):
        if "curvature" in w:
            weight = w["curvature"]
            st.insert([u, v], filtration=weight)

    st.make_filtration_non_decreasing()
    st.expansion(k)
    persistence_pairs = st.persistence(persistence_dim_max=True)

    diagrams = []

    for dimension in range(k + 1):
        diagram = [(c, d) for dim, (c, d) in persistence_pairs if dim == dimension]

        diagrams.append(diagram)

    return diagrams


def propagate_node_attribute_to_edges(G, attribute, pooling_fn=max):
    """Propagate a node attribute to edges.

    This function propagates a node attribute, such as the degree,
    to an edge attribute of the same name. This is done by calling
    a pooling function that condenses information of the attribute
    values of nodes incident on an edge.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Note that this graph will be changed **in place**.

    attribute : str
        Node attribute to use for the propagation procedure.

    pooling_fn : callable
        Function to pool node attribute information. Must be compatible
        with the node attribute type. The pooling function is called to
        summarize all node attributes that belong to an edge, i.e. only
        source and target node attributes.

        The pooling function must return a scalar value when provided
        with the source and target node of an edge. While other types
        of values are supported in principle, they will not result in
        graphs that are amenable to persistent homology calculations.
    """
    edge_attributes = dict()
    node_attributes = nx.get_node_attributes(G, attribute)

    for edge in G.edges(data=False):
        source, target = edge

        edge_attributes[edge] = pooling_fn(node_attributes[source], node_attributes[target])

    nx.set_edge_attributes(G, edge_attributes, name=attribute)


def propagate_edge_attribute_to_nodes(G, attribute, pooling_fn=np.sum):
    """Propagate an edge attribute to nodes.

    This function propagates an edge attribute, such as a curvature
    measurement, to a node attribute of the same name. This is done
    by calling a pooling function that condenses information of the
    attribute values of edges incident on a node.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Note that this graph will be changed **in place**.

    attribute : str
        Edge attribute to use for the propagation procedure.

    pooling_fn : callable
        Function to pool edge attribute information. Must be compatible
        with the edge attribute type. The pooling function is called to
        summarize all edge attributes that belong to a node, i.e. *all*
        attributes belonging to incident edges.
    """
    node_attributes = collections.defaultdict(list)

    for edge in G.edges(data=True):
        source, target, data = edge

        node_attributes[source].append(data[attribute])
        node_attributes[target].append(data[attribute])

    node_attributes = {node: pooling_fn(values) for node, values in node_attributes.items()}

    nx.set_node_attributes(G, node_attributes, name=attribute)


def make_node_filtration(G, edge_values, attribute_name="weight", use_min=True):
    """Create filtration based on edge values.

    This function takes a vector of edge values and assigns it to
    a graph in order to create a valid filtration. Note that this
    function creates both edge and vertex attributes. As a result
    of this operation, topological features can be calculated.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    edge_values : iterable
        Sequence of edge values. Depending on the `use_min` parameter,
        either the minimum of all edge values or the maximum of all edge
        values is assigned to a vertex.

    attribute_name : str
        Vertex attribute name for storing the values.

    use_min : bool
        If set, assigns each vertex the minimum of its neighbouring
        function values. Else, the maximum is assigned.

    Returns
    -------
    nx.Graph
        Copy of the input graph, with additional vertex attributes.
    """
    G = G.copy()

    vertex_values = collections.defaultdict(list)

    for edge, value in zip(G.edges(), edge_values):
        source, target = edge

        vertex_values[source].append(value)
        vertex_values[target].append(value)

    # this doesn't work if the graph isn't fully connected. I here set the curvature to be zero at a vertex in nodes but not edges
    for node in G.nodes():
        if node not in vertex_values:
            vertex_values[node].append(0)

    for v, values in vertex_values.items():
        if use_min:
            vertex_values[v] = np.min(values)
        else:
            vertex_values[v] = np.max(values)

    nx.set_node_attributes(G, vertex_values, attribute_name)
    nx.set_edge_attributes(
        G,
        # Create an in-line dictionary to assign the curvature values
        # properly to the edges.
        {e: v for e, v in zip(G.edges, edge_values)},
        attribute_name,
    )

    return G
