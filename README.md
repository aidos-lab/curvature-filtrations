# SCOTT

## **S**ynthesizing **C**urvature **O**perations and **T**opological **T**ools

[![arXiv](https://img.shields.io/badge/arXiv-2301.12906-b31b1b.svg)](https://arxiv.org/abs/2301.12906) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/CFGGME) ![GitHub](https://img.shields.io/github/license/aidos-lab/CFGGME)

`SCOTT` is a Python package for computing **curvature filtrations** for graphs and graph distributions. This repository accompanies our NeurIPS 2023 paper: [_Curvature Filtrations for Graph Generative Model Evaluation_](https://arxiv.org/abs/2301.12906).

Our method introduces a novel way to compare graph distributions, avoiding the limitations of Maximal Mean Discrepancy (MMD), which has known [drawbacks](https://arxiv.org/abs/2106.01098).

By combining **discrete curvature** on graphs with **persistent homology**, `SCOTT` provides expressive descriptors of graph sets that are:

- **Robust**
- **Stable**
- **Expressive**
- **Compatible with Statistical Testing**

The package is highly adaptable, offering several options for **user customization**, including different curvature computation methods and diverse metrics for comparing persistent homology outputs.

### **Cite Us**

If you find this package useful in your research, please consider citing:

```bibtex
@misc{southern2023curvature,
      title={Curvature Filtrations for Graph Generative Model Evaluation},
      author={Joshua Southern and Jeremy Wayland and Michael Bronstein and Bastian Rieck},
      year={2023},
      eprint={2301.12906},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Installation

### Using `pip`

Install `curvature-filtrations` via pip:

```bash
$ pip install curvature-filtrations
```

### Building from Source

Our dependencies are managed with [`poetry`](https://python-poetry.org), which can be installed with `pip install poetry`. To install from source:

1. Clone the repository

```bash
$ git clone https://github.com/aidos-lab/curvature-filtrations.git
```

1. Navigate to the directory

```bash
$ cd curvature-filtrations
```

3. Install dependencies

```bash
$ poetry install
```

# Quick Start

The example.py script demonstrates how to compute the distance between two graph distributions.

To use SCOTT with your own data, replace the example graph distributions with your own. Distributions should be lists of `networkx` graphs or single `networkx` graphs.

### Run our Example Script

```bash
python scripts/example.py
```

# Core Components

### KILT

`KILT` stands for: *K*rvature-*I*nformed *L*inks and *T*opology is an object that can compute curvature filtrations for single graphs.

```python
import networkx as nx
from scott import KILT,Comparator

G = nx.erdos_reyni(14,0.4)

kilt = KILT(measure="forman_curvature")

D = kilt.fit_transform(G)
print(f"Forman Curvature Filtration:")
print(f"Curvature Filtration Values:{kilt.curvature}")
print(D)
```

### Comparator

`Comparator` handles comparisons between graphs or graph distributions!

```python
import networkx as nx
from scott import KILT,Comparator

graph_dist1 = [nx.erdos_reyni(10,0.4) for _ in range(40)]
graph_dist2 = [nx.erdos_reyni(20,0.6) for _ in range(50)]

compare = Compare(measure="forman_curvature")

dist = compare.fit_transform(graph_dist1,graph_dist2,metric="image")

print(f"Distance between distributions measured by Forman Filtration: {dist}")
```
