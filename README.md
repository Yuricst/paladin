# `paladin` : python library for astrodynamics

| PyPI     | Documentation |
| -------- | ------------- |
| [![PyPI version](https://badge.fury.io/py/spacecraft-paladin.svg)](https://badge.fury.io/py/spacecraft-paladin)  | [![documentation workflow](https://github.com/Yuricst/paladin/actions/workflows/documentation.yml/badge.svg)](https://yuricst.github.io/paladin/)    |

!! Currently only tested/working with python 3.11 and numpy 1.X (due to restriction by pygsl) !!

## Installation

To install via pip

```bash
pip install spacecraft-paladin
```

To uninstall

```bash
pip uninstall spacecraft-paladin
```

## Environment setup

An example virtual environment for working with this library can be initialized using [`poetry`](https://python-poetry.org/) via

```bash
poetry install
pip install pygsl
```

Note: `pygsl` cannot be imported via `poetry` as it does not PEP 517 builds. 


## Overview

This package provides tools for conducting CR3BP and ephemeris level analysis in cislunar and deep-space environments. 

While the library is implemented in python, the majority of functionalities are powered by either [SPICE routines](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/index.html) or through numpy/scipy/numba implementations, resulting in (relatively) fast computations. 
For numerical integration of ODEs, `paladin` provides the option of using either `scipy` or `gsl`; the latter is recommended due to higher accuracy. 

Optimization is conducted by constructing problems as [`pygmo` udp's](https://esa.github.io/pygmo2/index.html), which can then be solved through a variety of compatible solvers, including IPOPT, SNOPT, or WORHP (the latter two requires licenses). 


## Dependencies

Developed for python 3.10 & 3.11.

Package requirements: 

- `numpy`, `matplotlib`, `numba`, `scipy`, `spiceypy`, `sympy`, `pygsl`

Optional:

- `pygmo`, `pygmo_plugins_nonfree` : required for running trajectory construction problems


## SPICE setup

Users are responsible for downloading [the generic SPICE kernels froom the NAIF website](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/). In addition, supplementary custom kernels specific to this module are stored in `paladin/assets/spice/`. The most commonly required kernels are:

- `naif0012.tls`
- `de440.bsp`
- `gm_de440.tpc` 


## GSL setup

See [pygsl setup notes](./notes/pygsl_setup.md)



## Capabilities

- [x] GSL event capability
- [x] Propagation in CR3BP
- [x] Propagation in N-body problem
- [x] Transition to full-ephemeris model
- [x] Helper methods for frame transformation


## Gallery

NRHO propagation

<p align="center">
  <img src="./plots/propagation_example_nrho.png" width="400" title="Propagation example">
</p>


