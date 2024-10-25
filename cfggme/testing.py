import networkx as nx
import numpy as np
import methods #seperate python file with the curvature functions


class Curvature:

    def __init__(self, ") -> None:
        self.method = method

    def fit(self, graph,method="ORC") -> np.array:
        # Compute curvature values for a graph
        self.curvature_fn = getattr(methods,method) # search through our methods to find the one specified by user
        self.curvature_values = self.curvature_fn(graph)
        pass

    def transform(self, graph, curvature_values) -> nx.Graph:
        # Add curvature features to a graph
        pass

    def fit_transform(self, graph) -> nx.Graph:
        self.fit(graph)
        self.transform(self.curvature_values)