import pytest
import networkx as nx
import numpy as np
from curvature_filtrations.topology.ph import GraphHomology, PersistenceDiagram


class TestDiagram:

    def test_create_object(self):
        diagram = PersistenceDiagram()
        assert type(diagram) == PersistenceDiagram

    def test_defaults(self):
        diagram = PersistenceDiagram()
        assert diagram.homology_dims == [0, 1]
        assert diagram.persistence_pts == None

    def test_set_persistence_pts(self, diagram_dict):
        diagram = PersistenceDiagram()
        assert diagram._persistence_pts == None
        # setting persistence points
        diagram.persistence_pts = diagram_dict
        assert diagram.persistence_pts == diagram_dict

    def test_get_pts_for_dim(self, diagram_dict):
        diagram = PersistenceDiagram()
        # set and get
        diagram.persistence_pts = diagram_dict
        print(diagram_dict[0])
        assert np.all(diagram.get_pts_for_dim(0) == diagram_dict[0])
