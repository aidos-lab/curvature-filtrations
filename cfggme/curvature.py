import networkx as nx
import numpy as np
import methods #seperate python file with the curvature functions

class Curvature:

    def __init__(self, method="forman_curvature", weight=None, alpha=0.0, prob_fn=None) -> None:
        """Defines the specifications for the desired method of computing curvature in a graph."""
        
        # Check that curvature method is supported
        assert method in ["forman_curvature", "ollivier_ricci_curvature", "resistance_curvature"]
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
        
        nx.set_edge_attributes(graph, curvature_values, name='curvature')
        return graph

    def fit_transform(self, graph) -> nx.Graph:
        """Computes the curvature values for the given graph according to the specifications of the Curvature object,
            and assigns them to their respective edges."""
        curvature_values = self.fit(graph)
        fitted_graph = self.transform(graph, curvature_values)
        return fitted_graph
        # streamlined: return self.transform(graph, self.fit(graph))

    # TODO: Topological section
    ## Topological Section (outcome is persistence diagram)
    def make_filtration():
        pass

    def generate_persistence_diagram():
        pass