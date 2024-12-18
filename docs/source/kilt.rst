scott.kilt
============

:code:`KILT` stands for **Krvature-Informed Links and Topology**, and is an object that can compute curvature filtrations for single graphs.

Example of KILT usage: ::

   import networkx as nx
   from scott import KILT,Comparator

   G = nx.erdos_reyni(14,0.4)

   kilt = KILT(measure="forman_curvature")

   D = kilt.fit_transform(G)
   print(f"Forman Curvature Filtration:")
   print(f"Curvature Filtration Values:{kilt.curvature}")
   print(D)



.. automodule:: scott.kilt
    :members: