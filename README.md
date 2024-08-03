# NeuroLISP
`neurolisp` includes an implementation of NeuroLISP, a programmable neural network that implements an interpreter for a dialect of the LISP programming language.

Davis, G.P., Katz, G.E., Gentili, R.J., Reggia, J.A. NeuroLISP: High-level symbolic programming with attractor neural networks. Neural Networks 146, 200-219 (2022).
[publication](https://doi.org/10.1016/j.neunet.2021.11.009)


## Requirements


* [numpy](http://www.numpy.org/) is required for the base implementation
* [pycuda](http://pypi.org/project/pycuda/) is required to run on GPUs
* [networkx](http://networkx.org/) is required to generate unification test cases
* [matplotlib](http://matplotlib.org/) is required to use built-in plotting of test data

## Installation

1. [Clone or download](https://help.github.com/articles/cloning-a-repository/) this repository into a directory of your choice.
2. Add the `src` sub-directory to your [PYTHONPATH](https://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH).

## Basic Usage

The ``build_neurolisp.py`` script contains functions for constructing a NeuroLISP instance and testing it with a LISP program.  Test cases of varying complexity are contained in ``test_neurolisp.py``, while the experimental tests used in the NeuroLISP paper are contained in ``exp_neurolisp.py``.  Here we describe a simple test that illustrates usage.  This code is contained in ``readme.py``.

Import the functions for testing NeuroLISP and preprocessing code:

```
from build_neurolisp import test, preprocess
```

Write a LISP function (see paper for supported operators):

```
code = '''
    (defun rev-helper (pre post)
        (if pre
            (rev-helper
                (cdr pre)
                (cons (car pre) post))
            post))
    (defun reverse (x) (rev-helper x NIL))

    (reverse (quote (a b c d e)))
'''
```

Preprocess code into a sequence of input tokens:

```
inputs = preprocess(code)
```

Run the test:

```
test(inputs=inputs,
    t=10000000,
    verbose=True,
    capacity = {
        "mem": 256+32,
        "lex": 64,
        "bind": 128,
        "stack": 128,
        "data_stack": 32,
    },
    ctx_lam = {
        "mem_ctx" : 0.25,
        "bind_ctx" : 0.125,
    })
```

The ``t`` parameter indicates the maximum number of timesteps to run before timeout, and ``verbose`` indicates whether to print verbose output during testing. The ``capacity`` parameter indicates the memory capacity of various regions. For stack regions, this is the number of stack frames. For other regions, this determines the number of neurons in the region, which is computed as ``16*cap``, as per the empirical memory capacity of attractor networks.
