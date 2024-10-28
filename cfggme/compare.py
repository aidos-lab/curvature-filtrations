import networkx as nx
import numpy as np
from curvature import Curvature
import topology

class GraphComparator:
    def __init__(self, graph1, graph2, curvature=Curvature()) -> None:
        """Class for comparing the curvature between two graphs."""
        self.graph1 = graph1
        self.graph2 = graph2
        self.curv = curvature

    # placeholder distance, TODO: remove
    def calc_total_curv_diff(self):
        curv1 = self.curv.fit(self.graph1)
        curv2 = self.curv.fit(self.graph2)
        return sum(curv1) - sum(curv2)
    
    # distance between persistance landscapes
    def curvature_filtration_distance(self): #TODO: do I need to put defaults here??
        """Calculates the curvature filtration distance between two graphs. (Note: Not two DISTRIBUTIONS of graphs)."""
        landscape1 = self.curv.make_landscape(self.graph1)
        landscape2 = self.curv.make_landscape(self.graph2)

        return np.linalg.norm(np.array(landscape1) - np.array(landscape2))