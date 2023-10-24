"""Various utility functions and classes."""

import collections
import multiprocessing
import warnings

import gudhi as gd
import gudhi.representations
import gudhi.wasserstein
import networkx as nx
import numpy as np
from joblib import Parallel, delayed


def create_landscapes_gudhi(curvature_fn, graphs, hom_deg=2, exact=True):
    """Calculates the persistence landscapes of a set of graphs.
    Parameters
    ----------
    graphs : list[networkx.Graphs]
        A list of Networkx graphs.
    curvature_fn : function
        A function which takes the networkx graph as input and outputs an np.array of curvature values.
    hom_deg : int
        The max homology dimension to consider.

    exact : boolean
        Whether to calculate the landscapes exactly or with approximation in the Gudhi package

    """

    def landscape_iterator(graph, exact=True):
        curvature = curvature_fn(graph)
        curvature = {e: c for e, c in zip(graph.edges(), curvature)}
        nx.set_edge_attributes(graph, curvature, "curvature")

        propagate_edge_attribute_to_nodes(graph, "curvature", pooling_fn=lambda x: -1)

        diagrams = calculate_persistent_homology(graph, k=2)

        if diagrams[1]:
            p_diagrams = np.concatenate(
                (np.array(diagrams[0]), np.array(diagrams[1])), axis=0
            )
        else:
            p_diagrams = np.array(diagrams[0])

        p_diagrams[
            p_diagrams == np.inf
        ] = 1000  # this is annoying but I have to remove inf values somehow

        LS = gd.representations.Landscape(resolution=1000)
        landscape = LS.fit_transform([p_diagrams])

        return landscape

    # Embarassingly Parallel Joblib Loops
    n_cpu = multiprocessing.cpu_count()
    all_landscapes = Parallel(n_jobs=n_cpu, prefer="threads")(
        delayed(landscape_iterator)(G, exact) for G in graphs
    )
    landscapes = [x for x in all_landscapes if x is not None]
    return landscapes


def average_landscape(landscapes, exact=True):
    assert exact, "Only exact landsape computations supported at the moment."

    sum_ = landscapes[0]
    N = len(landscapes)
    for landscape in landscapes[1:]:
        sum_ += landscape
    return sum_.__truediv__(N)


def curvature_filtration_distance(
    graphs1, graphs2, curvature_fn, hom_deg=2, exact=True
):
    """Calculates the curvature filtration distance between two distributions of graphs.
    Parameters
    ----------
    graphs1 : list[networkx.Graphs]
        A list of Networkx graphs.
    graphs2 : list[networkx.Graphs]
        A list of Networkx graphs.
    curvature_fn : function
        A function which takes the networkx graph as input and outputs an np.array of curvature values.
    hom_deg : int
        The max homology dimension to consider.

    exact : boolean
        Whether to calculate the landscapes exactly or with approximation in the Gudhi package

    """
    landscapes1 = create_landscapes_gudhi(
        curvature_fn, graphs1, hom_deg=hom_deg, exact=exact
    )

    landscapes2 = create_landscapes_gudhi(
        curvature_fn, graphs2, hom_deg=hom_deg, exact=exact
    )

    avg1, avg2 = average_landscape(landscapes1, exact), average_landscape(
        landscapes2, exact
    )

    return np.linalg.norm(np.array(avg1) - np.array(avg2))


def calculate_persistent_homology(G, k=3):
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

        edge_attributes[edge] = pooling_fn(
            node_attributes[source], node_attributes[target]
        )

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

    node_attributes = {
        node: pooling_fn(values) for node, values in node_attributes.items()
    }

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


class UnionFind:
    """An implementation of a Union--Find class.

    The class performs path compression by default. It uses integers for
    storing one disjoint set, assuming that vertices are zero-indexed.
    """

    def __init__(self, n_vertices):
        """Initialise an empty Union--Find data structure.

        Creates a new Union--Find data structure for a given number of
        vertices. Vertex indices are assumed to range from `0` to
        `n_vertices`.

        Parameters
        ----------
        n_vertices:
            Number of vertices
        """
        self._parent = [x for x in range(n_vertices)]

    def find(self, u):
        """Find and return the parent of `u` with respect to the hierarchy.

        Parameters
        ----------
        u:
            Vertex whose parent is looked up

        Returns
        -------
        Component the vertex belongs to.
        """
        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        """Merge vertex `u` into the component of vertex `v`.

        Performs a `merge()` operation. Note the asymmetry of this
        operation, as vertex `u` will be  merged into the connected
        component of `v`.

        Parameters
        ----------
        u:
            Source connected component

        v:
            Target connected component
        """
        # There is no need to adjust anything if, by some fluke, we
        # merge ourselves into our parent component.
        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        """Generate roots.

        Generator expression for returning roots, i.e. components that
        are their own parents.

        Returns
        -------
        Yields each root vertex.
        """
        # We assume that vertices are numbered contiguously from zero to
        # `n_vertices`. This simplifies identifying a vertex here.
        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex

    def get_component(self, root):
        """Get vertices belonging to a specific component.

        Parameters
        ----------
        root:
            Root of the specified connected component.

        Returns
        -------
        List of vertices in said connected component.
        """
        return [v for v, p in enumerate(self._parent) if p == root]
