import networkx as nx
import numpy as np
import cfggme.methods as methods  # seperate python file with the curvature functions
import cfggme.topology as topology
import gudhi as gd


class Curvature:

    def __init__(
        self, method="forman_curvature", weight=None, alpha=0.0, prob_fn=None
    ) -> None:
        """Defines the specifications for the desired method of computing curvature in a graph."""

        # Check that curvature method is supported
        assert method in [
            "forman_curvature",
            "ollivier_ricci_curvature",
            "resistance_curvature",
        ]
        self.method = method

        self.weight = weight
        self.alpha = alpha
        self.prob_fn = prob_fn

    ## Geometric Section (i.e. computing curvature)
    def fit(self, graph) -> np.array:
        """Computes curvature values for the given graph according to the specifications of the Curvature object."""
        # Search through our methods to find the one specified by user
        curvature_fn = getattr(methods, self.method)
        # Check that function is callable
        # assert callable(curvature_fn(graph))

        if self.method == "ollivier_ricci_curvature":
            # Ollivier Ricci method supports extra inputs
            return curvature_fn(graph, self.alpha, self.weight, self.prob_fn)
        else:
            # Forman and Resistance methods only require graph and optional weight
            return curvature_fn(graph, self.weight)

    def transform(self, graph, curvature_values) -> nx.Graph:
        """Assigns the values of the given curvature_values np.array to the respective edges of the given graph."""
        nx.set_edge_attributes(graph, curvature_values, name="curvature")
        return graph

    def fit_transform(self, graph) -> nx.Graph:
        """Computes the curvature values for the given graph according to the specifications of the Curvature object,
        and assigns them to their respective edges."""
        curvature_values = self.fit(graph)
        fitted_graph = self.transform(graph, curvature_values)
        return fitted_graph
        # streamlined: return self.transform(graph, self.fit(graph))

    ## Topological Section (outcome is persistence diagram)
    def make_landscape(self, graph, exact=True):
        curvature = self.fit(graph)
        curvature = {e: c for e, c in zip(graph.edges(), curvature)}
        nx.set_edge_attributes(graph, curvature, "curvature")

        topology.propagate_edge_attribute_to_nodes(
            graph, "curvature", pooling_fn=lambda x: -1
        )
        diagrams = topology.calculate_persistent_homology(graph, k=2)

        if diagrams[1]:
            p_diagrams = np.concatenate(
                (np.array(diagrams[0]), np.array(diagrams[1])), axis=0
            )
        else:
            p_diagrams = np.array(diagrams[0])

        p_diagrams[p_diagrams == np.inf] = (
            1000  # this is annoying but I have to remove inf values somehow
        )

        LS = gd.representations.Landscape(resolution=1000)
        landscape = LS.fit_transform([p_diagrams])

        return landscape

    def __str__(self) -> str:
        """Return a string representation of the Curvature and any custom attributes."""
        name = f"Method: {self.method}"
        if self.weight != None:
            name += f"\nCustom Weight Attribute: {self.weight}"
        if self.alpha != 0.0:
            name += f"\nCustom Alpha: {self.alpha}"
        if self.prob_fn != None:
            name += f"\nCustom Probability Function: {self.prob_fn}"
        return name
