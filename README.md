## PyTorch (GPU) implementation of Higher Order Singular Value Decomposition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/whistlebee/pytorch-hosvd/blob/master/notebooks/experiments.ipynb)

Has:
* sequential truncation [1]
* randomized svd [2]

Have a look at the [notebook](experiments.ipynb) for examples.

![](images/comparison.png?raw=true)

[1] Vannieuwenhoven, Nick, Raf Vandebril, and Karl Meerbergen. "A new truncation strategy for the higher-order singular value decomposition." SIAM Journal on Scientific Computing 34.2 (2012): A1027-A1052.

[2] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM review 53.2 (2011): 217-288.
