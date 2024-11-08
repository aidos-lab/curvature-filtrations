import pytest
import networkx as nx
import numpy as np
from curvature_filtrations.topology.ph import GraphHomology, PersistenceDiagram
from curvature_filtrations.kilt import KILT


class TestDiagram:

    def test_create_object(self):
        diagram = PersistenceDiagram()
        assert type(diagram) == PersistenceDiagram

    def test_defaults(self):
        diagram = PersistenceDiagram()
        assert diagram.homology_dims == [0, 1]
        assert diagram.persistence_pts == None

    def test_set_persistence_pts(self, dummy_diagram):
        diagram = PersistenceDiagram()
        assert diagram._persistence_pts == None
        # setting persistence points
        diagram.persistence_pts = dummy_diagram
        assert diagram.persistence_pts == dummy_diagram

    def test_get_pts_for_dim(self, dummy_diagram):
        diagram = PersistenceDiagram()
        # set and get
        diagram.persistence_pts = dummy_diagram
        print(dummy_diagram[0])
        assert np.all(diagram.get_pts_for_dim(0) == dummy_diagram[0])

    def test_ph_calc(self, graph):
        klt = KILT()
        ph = klt.fit_transform(graph)
        assert type(ph) == PersistenceDiagram
