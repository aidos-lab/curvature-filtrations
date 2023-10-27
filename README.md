# cfggme: <u>C</u>urvature <u>F</u>iltrations <u>G</u>raph <u>G</u>enerative <u>M</u>odel <u>E</u>valuation

[![arXiv](https://img.shields.io/badge/arXiv-2301.12906-b31b1b.svg)](https://arxiv.org/abs/2301.12906) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/CFGGME) ![GitHub](https://img.shields.io/github/license/aidos-lab/CFGGME)

This is the official repository for the NeurIPS 2023 paper: [Curvature Filtrations for Graph Generative Model Evaluation](https://arxiv.org/abs/2301.12906).


We provide a new method for comparing graph distributions that _does not_ rely on Maximal Mean Discrepancy (MMD), which has been shown to have some [drawbacks](https://arxiv.org/abs/2106.01098). Our method combines **discrete curvature** on graphs and **persistent homology** to build expressive descriptors of sets
of graphs that are _robust_, _stable_, _expressive_ and support _statistical tests_. When using our measures in practice, we highly reccomend selecting a filtration
based on **Ollivier Ricci** curvature.

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

# Dependencies

Dependencies are managed using `poetry.` To setup the environment,
please run `poetry install` from the main directory (assuming the user
already has installed `poetry`).

# Running CFGGME

The `example.py` shows an example of how we generate a distance between two distributions of graphs.  We assume that each distribution of graphs is stored as a list of `networkx` graphs. To use our code for comparing your own distributions of graphs, it is as easy as substituting the two lists with your own lists of graphs.

The `sr.py` file recreates the strongly regular graph experiments shown in the paper. 


