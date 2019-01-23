# SMAC v3 Project

This is a branch of automl/SMAC3 of v0.8.0 for study and development purpose. 
The ideas are provided by Dr. Daning Cheng and implemented by myself (Hanping Zhang).
Only raw results will be shown until the final decision of the conference come out.
##### Please see Usage section and Modification section for more details.

Copyright (C) 2016-2018  [ML4AAD Group](http://www.ml4aad.org/)

__Attention__: This package is a re-implementation of the original SMAC tool
(see reference below).
However, the reimplementation slightly differs from the original SMAC.
For comparisons against the original SMAC, we refer to a stable release of SMAC (v2) in Java
which can be found [here](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/).

The documentation can be found [here](https://automl.github.io/SMAC3/).

Status for master branch:

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=master)](https://travis-ci.org/automl/SMAC3)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/58f47a4bd25e45c9a4901ebca68118ff?branch=master)](https://www.codacy.com/app/automl/SMAC3?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=automl/SMAC3&amp;utm_campaign=Badge_Grade)
[![codecov Status](https://codecov.io/gh/automl/SMAC3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

Status for development branch

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=development)](https://travis-ci.org/automl/SMAC3)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/58f47a4bd25e45c9a4901ebca68118ff?branch=development)](https://www.codacy.com/app/automl/SMAC3?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=automl/SMAC3&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/automl/SMAC3/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

# OVERVIEW

SMAC is a tool for algorithm configuration to optimize the parameters of
arbitrary algorithms across a set of instances. This also includes
hyperparameter optimization of ML algorithms. The main core consists of
Bayesian Optimization in combination with a aggressive racing mechanism to
efficiently decide which of two configuration performs better.

For a detailed description of its main idea,
we refer to

    Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
    Sequential Model-Based Optimization for General Algorithm Configuration
    In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)


SMAC v3 is written in Python3 and continuously tested with python3.5 and
python3.6. Its [Random Forest](https://github.com/automl/random_forest_run)
is written in C++.

# Installation

## Requirements

Besides the listed requirements (see `requirements.txt`), the random forest
used in SMAC3 requires SWIG (>= 3.0).

```apt-get install swig```


## Installation via pip

SMAC3 is available on pipy.

```pip install smac```

## Manual Installation

```
git clone https://github.com/automl/SMAC3.git && cd SMAC3
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## Installation in Anaconda

If you use Anaconda as your Python environment, you have to install three
packages **before** you can install SMAC:

```conda install gxx_linux-64 gcc_linux-64 swig```

# License

This program is free software: you can redistribute it and/or modify
it under the terms of the 3-clause BSD license (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the 3-clause BSD license
along with this program (see LICENSE file).
If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# USAGE

The usage of SMAC v3 is mainly the same as provided with [SMAC v2.08](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/v2.08.00/manual.pdf).
It supports the same parameter configuration space syntax
(except for extended forbidden constraints) and interface to
target algorithms.

First, please make sure you setup all the required python packages of SMAC3, including swig3.

# Examples

See examples/

  * examples/rosenbrock.py - example on how to optimize a Python function
  * examples/spear_qcp/run.sh - example on how to optimize the SAT solver Spear
    on a set of SAT formulas

# Contact

SMAC3 is developed by the [ML4AAD Group of the University of Freiburg](http://www.ml4aad.org/).

If you found a bug, please report to https://github.com/automl/SMAC3

# Modification

Following files have been modified:

### data/

* dataset_1049_pc4.csv <br>
We use pc4, real-sim and rcv1 for testing.

### results/

* fitting_loss_in_smbo/ <br>
Loss curves predicted by gradient-based GPR in each iteration.

* gpr_fitting_graph/ <br>
Comparison of hyperboloid fitting of normal GPR and gradient-based GPR.

* smac_comparison/ <br>
A simple comparison between normal SMAC with RF and our gradient-based GPR.

### smac/epm/

* bayes_opt/ <br>
Forked from fmfn/BayesianOptimization. Used for GP based SMAC.

* gaussian_gradient_epm.py <br>
EPM for gradient-based Gaussian Process Regressor.

* gaussian_process/ <br>
Including self-implemented Gaussian Kernels and the gradient-based Gaussian 
Process Regressor.

* hoag/ <br>
Forked from fabianp/hoag. Used to calculate gradients of the loss.

### smac/facade/

* oursmac_facade.py <br>
A simple class to initialize our modified SMAC.

* smac_facade.py <br>
Add parameters hoag, server and bayesian_optimization 
to SMAC class. <br>
Usage: <br>
hoag: Default value is None. It should be set to a subclass that 
inherit AbstractHOAG. The gradient-based GPR will be invoked if this value is
 not None. <br>
server: Default value is None. It should be set to a PS_SMAC Server 
instance. The PS_SMAC will be invoked if this value is not None. <br>
bayesian_optimization: Default value is False. If this flag is set to True, 
we will use bayesian_optimization instead of original RF based SMAC; However,
 if the hoag parameter is not None, this flag will be ignored.

### smac/optimizer/

* smbo.py <br>
The main loop of SMBO process. Add PS-Lite server in the main BO loop and 
hoag, gradient-based GPR and bayesian_optimization in the choose_next function.

### smac/pssmac/

See README.md in smac/pssmac/ for usage.
* facade/ <br>
Facades of the PS-Lite nodes, including the abstract_facade and related 
scheduler_facade, server_facade and worker_facade. Used to create complete
scheduler/server/worker process.

* ps/ <br>
Implementations of parallel SMAC based on PS-Lite, which is an asynchronous 
parallel process based on Parameter Server architect. <br>
ps_smac.cc is a C++ implementation of server/worker nodes. abstract_ps and 
its subclasses server.py and worker.py define the python nodes which pass the
 hyperparameters and runhistory to the PS-Lite using pipes.

* tae/ <br>
An abstract class for ta functions (model) and an instance for 
LogisticRegression.

### smac/runhistory/

* runhistory.py <br>
Add a util function get_history_for_config to the runhistory class.

### smac/stats/

* stats.py <br>
Modify the cutoff inside the class to avoid running forever.

### smac/tae/

* execute_ta_customized.py <br>
Temporary ta function for LogisticRegression.

### smac/utils/

* libsvm2sparse.py <br>
A util function to parse libsvm format data to sparse matrix.

* util_funcs.py <br>
Add a util function remove_same_values in order to removed repeated lines in 
the runhistory passed from the main BO loop to choose_next.
