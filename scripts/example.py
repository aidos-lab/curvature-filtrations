"Example: Computing Distance between 2 ER Graph Distributions"

import networkx as nx
from scott.compare import Comparator

# Defining the two graph distributions that will be compared.
# Input your own graph distributions (lists of networkx graphs) here. Individual networkx graphs are also supported.

graph_distribution1 = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]
graph_distribution2 = [nx.erdos_renyi_graph(20, 0.2) for _ in range(10)]

#  ╭──────────────────────────────────────────────────────────╮
#  │ Customizing Curvature Measure                            │
#  ╰──────────────────────────────────────────────────────────╯

# The curvature measure can be customized via the 'measure' parameter when initializing an instance of the Comparator class.
# For additional customization options, please see compare.py.

# Using Forman-Ricci Curvature (Default)
comp_forman = Comparator()
print(
    "Distance between distributions with Forman-Ricci Curvature (default): {}".format(
        comp_forman.fit_transform(graph_distribution1, graph_distribution2)
    )
)

# Using Ollivier-Ricci Curvature
comp_orc = Comparator(measure="ollivier_ricci_curvature")
print(
    "Distance between distributions with Ollivier-Ricci Curvature: {}".format(
        comp_orc.fit_transform(graph_distribution1, graph_distribution2)
    )
)

# Using Resistance Curvature
comp_resist = Comparator(measure="resistance_curvature")
print(
    "Distance between distributions with Resistance Curvature: {}".format(
        comp_resist.fit_transform(graph_distribution1, graph_distribution2)
    )
)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Customizing Distance Metric                              │
#  ╰──────────────────────────────────────────────────────────╯

# All of the above distances were calculated with the default distance metric, i.e. using a persistence landscape vectorization.
# This part of the computation can be customized by the 'metric' parameter in the fit_transform method, and paired with any of the curvature measures above.
# In the interest of brevity, all the examples below will use the default curvature measure, Forman-Ricci.

# Using the Persistence Landscape Vectorization (Default)
print(
    "Distance between distributions with persistence landscape vectorization (default): {}".format(
        comp_forman.fit_transform(graph_distribution1, graph_distribution2)
    )
)

# Using the Persistence Image Vectorization
print(
    "Distance between distributions with persistence image vectorization: {}".format(
        comp_forman.fit_transform(graph_distribution1, graph_distribution2, metric="image")
    )
)
