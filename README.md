# Distributed MPC for PWA Systems Based on Switching ADMM

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/hybrid-vehicle-platoon/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in example 2 of [Distributed MPC for PWA Systems Based on Switching ADMM](https://arxiv.org/abs/2404.16712) submitted to [IEEE Transactions on Automatic Control](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9).

In this work, we propose a benchmark control problem for evaluating distributed hybrid model predictive controllers. The benchmark problem is the control of a platoon of vehicles, with the vehicle dynamics modelled as a hybrid system. We present two modelling approaches and evaluate five existing hybrid model predictive controllers on the benchmark.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{
    mallick2024distributed,
    title={Distributed MPC for PWA Systems Based on Switching ADMM}, 
    author={Samuel Mallick and Azita Dabiri and Bart De Schutter},
    year={2024},
    eprint={2404.16712},
    archivePrefix={arXiv}
}
```

---

## Installation

The code was created with `Python 3.11`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/hybrid-vehicle-platoon
cd hybrid-vehicle-platoon
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`misc`** contains auxilary classes and common controller parameters.
- **`mpcs`** contains base classes for model predictive controller formulations. 
- **`models.py`** contains the vehicle models used for prediction and simulation.
- **`env.py`** contains the environment for simulating the platoon.
- **`fleet_{cent_mld, g_admm, seq_mld, naive_admm}.py`** are the controllers used in Distributed MPC for PWA Systems Based on Switching ADMM.
- **`all other files`** are for data processing, or other projects that consider the platoon problem.
- **`data`** contains '.pkl' files for data used in Distributed MPC for PWA Systems Based on Switching ADMM.
```

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/hybrid-vehicle-platoon/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2023 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “hybrid-vehicle-platoon” (A Comparison Benchmark for Distributed Hybrid MPC Control Methods: Distributed Vehicle Platooning) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
