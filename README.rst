
Benchopt benchmark for multitask Lasso
======================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solvers of the **multitask Lasso** problem:


$$\\min_{W} \\frac{1}{2} \\Vert Y - XW \\Vert^2_2 + \\lambda  \\sum_{j=1}^p \\Vert W_{j:} $$


where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features,
$q$ (or ``n_tasks``) stands for the number of tasks and


$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad W \\in \\mathbb{R}^{p \\times q}$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_multitask_lasso
   $ benchopt run benchmark_multitask_lasso

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_multitask_lasso -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_multitask_lasso/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_multitask_lasso/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
