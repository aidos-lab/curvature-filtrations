# scott: <u>S</u>tudying <u>C</u>urvature <u>O</u>perations and <u>T</u>opological <u>T</u>oolkit

[![arXiv](https://img.shields.io/badge/arXiv-2301.12906-b31b1b.svg)](https://arxiv.org/abs/2301.12906) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/CFGGME) ![GitHub](https://img.shields.io/github/license/aidos-lab/CFGGME)

This is the official repository for the NeurIPS 2023 paper: [Curvature Filtrations for Graph Generative Model Evaluation](https://arxiv.org/abs/2301.12906).


We provide a new method for comparing graph distributions that _does not_ rely on Maximal Mean Discrepancy (MMD), which has been shown to have some [drawbacks](https://arxiv.org/abs/2106.01098).

Our method combines **discrete curvature** on graphs and **persistent homology** to build expressive descriptors of sets of graphs that are _robust_, _stable_, _expressive_ and support _statistical tests_.

The package is meant to be adaptable, and features several options for **user customization**. In particular, we support several methods for calculating curvature, as well as several different metrics / expressive descriptors for computing the distance between persistent homology outputs.

Please consider citing our work! 

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

# Dependencies & Setup

Dependencies are managed using the [`poetry`](https://python-poetry.org) package manager.

With poetry installed and an active virtual environment, run the following command from the main directory to download the necessary dependencies:

```python
$ poetry install
```


# Quick Start with SCOTT

The `example.py` file demonstrates how we generate a distance between two distributions of graphs. 

To use our code for comparing your own distributions of graphs, simply substitue the two lists with your own graph distributions. We assume that each distribution of graphs is stored as a list of `networkx` graphs, or simply as a single `networkx` graph.

Several basic customization options are demonstrated in `example.py`. Please see our `customize_compare.ipynb` tutorial if more customization is desired.


# SCOTT Tutorials

For more a more in-depth walkthrough of customization options and the intermediary functionalities supported by SCOTT objects, please see `/notebooks`. There are two main tutorials:

(1) **Customizing how your comparison is executed:** `custom_compare.ipynb`

_Read this section if:_ Your primary goal is to find the distance between your graph distributions, but you are looking for additional ways to customize the curvature and distance measures used.

(2) **Breakdown of intermediate functionalities:** `bagpipeline.ipynb`

_Read this section if:_ You want to better understand the underlying workflow of this process and/or are interested in the output from intermediate steps in the process.

Functionalities demonstrated in this tutorial include:
- Calculating curvature for one graph or graph distribution
- Executing a curvature filtration to produce a persistence diagram
- Converting persistence diagrams into a topological descriptor (e.g. persistence landscape, persistence image)
- Computing the distance between topological descriptors


