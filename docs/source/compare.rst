scott.compare
=============

The :code:`Comparator` class handles comparisons between graphs or graph distributions.

Example of Comparator usage: ::

   import networkx as nx
   from scott import KILT, Comparator

   graph_dist1 = [nx.erdos_reyni(10,0.4) for _ in range(40)]
   graph_dist2 = [nx.erdos_reyni(20,0.6) for _ in range(50)]

   compare = Compare(measure="forman_curvature")

   dist = compare.fit_transform(graph_dist1,graph_dist2,metric="image")

   print(f"Distance between distributions measured by Forman Filtration: {dist}")


.. automodule:: scott.compare
    :members: