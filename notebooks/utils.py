# Importing necessary packages
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import gudhi as gd

from scott.kilt import KILT
from scott.topology.representations import PersistenceDiagram, PersistenceLandscape

#  ╭──────────────────────────────────────────────────────────╮
#  │ Grabbing Dummy Objects                                   │
#  ╰──────────────────────────────────────────────────────────╯


def get_graphs():
    """Returns 2 Erdos-Renyi graphs with 100 nodes."""
    return nx.erdos_renyi_graph(100, 0.1), nx.erdos_renyi_graph(100, 0.1)


def get_distributions():
    """Returns 2 distributions of 100-node Erdos-Renyi graphs with length 8 and 10."""
    return [nx.erdos_renyi_graph(100, np.random.rand()) for _ in range(8)], [
        nx.erdos_renyi_graph(100, np.random.rand()) for _ in range(10)
    ]


def get_demo_graph():
    """Returns a demo graph with Forman curvature."""
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(range(8))

    # Add edges
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 5),
        (4, 5),
        (5, 6),
        (3, 7),
    ]
    G.add_edges_from(edges)

    kilt = KILT()
    kilt.fit(G)
    return kilt.G


def get_demo_graph2():
    """Returns a demo graph with Forman curvature."""
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(range(8))

    # Add edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1), (7, 2)]
    G.add_edges_from(edges)

    kilt = KILT()
    kilt.fit(G)
    return kilt.G


#  ╭──────────────────────────────────────────────────────────╮
#  │ Plotting                                                 │
#  ╰──────────────────────────────────────────────────────────╯


def plot_graph(graph, title=""):
    """Plots a single NetworkX graph."""
    plt.figure(figsize=(2, 2))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos, node_color="darkblue", node_size=30)
    plt.title(title)
    plt.show()


def plot_graphs(graph1, graph2, title1="", title2=""):
    """Plots two given NetworkX graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    nx.draw(
        graph1,
        ax=axes[0],
        node_size=30,
        node_color="darkblue",
        edge_color="gray",
        alpha=0.6,
    )
    axes[0].set_title(title1)
    nx.draw(graph2, ax=axes[1], node_size=30, node_color="orange", edge_color="gray", alpha=0.6)
    axes[1].set_title(title2)
    plt.tight_layout()
    plt.show()


def plot_distributions(dist1, dist2):
    """Plots two lists of NetworkX graphs."""
    fig1, axes = plt.subplots(1, len(dist1), figsize=(24, 6))
    for idx, graph in enumerate(dist1):
        nx.draw(
            graph, ax=axes[idx], node_size=30, node_color="darkblue", edge_color="gray", alpha=0.6
        )
    axes[0].set_title("Graph Distribution 1")
    plt.tight_layout()
    plt.show()

    fig2, axes = plt.subplots(1, len(dist2), figsize=(24, 6))
    for idx, graph in enumerate(dist2):
        nx.draw(
            graph, ax=axes[idx], node_size=30, node_color="orange", edge_color="gray", alpha=0.6
        )
    axes[0].set_title("Graph Distribution 2")
    plt.tight_layout()
    plt.show()


def plot_curvature(graph, title=""):
    """Plots a graph with edge curvature values."""
    plt.figure(figsize=(2, 2))
    # Extract edge weights (attribute values)
    curvature = nx.get_edge_attributes(graph, "forman_curvature")
    curv_vals = list(curvature.values())
    # Define positions for nodes
    pos = nx.spring_layout(graph)
    # Normalize the weights for the colormap
    vmin = min(curvature.values())
    vmax = max(curvature.values())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Define a colormap (e.g., "viridis")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")
    # Draw the graph
    nx.draw(graph, pos, node_size=35, node_color="black", edge_color=curv_vals, width=2)
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(curv_vals)
    plt.colorbar(sm, ax=plt.gca(), label="Curvature Scale")
    plt.title(title)

    # Show the graph
    plt.show()


def plot_diagram(diagram: PersistenceDiagram, title=""):
    """Plots a PersistenceDiagram object using gudhi."""
    # Converting to format that gudhi accepts: [(dim, (birth, death)), ...)]
    persistence = []
    for dim in diagram.homology_dims:
        for point in diagram.get_pts_for_dim(dim):
            persistence.append((dim, (point[0], point[1])))
    gd.plot_persistence_diagram(persistence)


def plot_landscape(landscape: PersistenceLandscape):
    """Plots the first few landscape functions of H0 for a PersistenceLandscape object."""
    # Get x-values (sampled points)
    x_values = np.linspace(0, 2, 1000)

    # Plot the first few landscape functions
    plt.figure(figsize=(10, 6))
    data0 = landscape.get_fns_for_dim(0)
    num_fns = landscape.num_functions
    for fn_num in range(0, int(num_fns)):
        start_idx = int(fn_num * 1000)
        indices = range(start_idx, start_idx + 1000)
        l = data0[indices]
        plt.plot(x_values, l, label=f"λ{fn_num + 1}")
    plt.title("Persistence Landscape for H0")
    plt.xlabel("x")
    plt.ylabel("λ(x)")
    plt.legend()
    plt.show()
