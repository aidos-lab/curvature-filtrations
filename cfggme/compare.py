import networkx as nx
import numpy as np
import curvature

class GraphComparator:
    def __init__(self, graph1, graph2, curvature=curvature.Curvature()) -> None:
        """Class for comparing the curvature between two graphs."""
        #TODO: Decide if we want to add method 2 in order to compare curvature (assuming graph1==graph2)
        
        self.graph1 = graph1
        self.graph2 = graph2
        self.curv = curvature

    def calc_total_curv_diff(self):
        curv1 = self.curv.fit(self.graph1)
        curv2 = self.curv.fit(self.graph2)
        return sum(curv1 - curv2)
        