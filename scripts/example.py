"Example: Computing Distance between 2 ER Graph Distributions"
import networkx as nx
from curvature import forman_curvature, ollivier_ricci_curvature, resistance_curvature
from utils import curvature_filtration_distance

graph_distribution1 = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]
graph_distribution2 = [nx.erdos_renyi_graph(20, 0.2) for _ in range(10)]

print(
    "distance between distributions with Forman-Ricci Curvature: {}".format(
        curvature_filtration_distance(
            graph_distribution1, graph_distribution2, forman_curvature
        )
    )
)

print(
    "distance between distributions with Ollivier-Ricci Curvature: {}".format(
        curvature_filtration_distance(
            graph_distribution1, graph_distribution2, ollivier_ricci_curvature
        )
    )
)

print(
    "distance between distributions with Resistance Curvature: {}".format(
        curvature_filtration_distance(
            graph_distribution1, graph_distribution2, resistance_curvature
        )
    )
)
