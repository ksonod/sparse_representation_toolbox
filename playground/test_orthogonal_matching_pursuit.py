# Orthogonal matching pursuit
# Michael Elad, Sparse and Redundant Representations, Chapter 3, page 36-37, 2010

import numpy as np
from algorithm.pursuit import GreedyAlgorithm
from sklearn.linear_model import OrthogonalMatchingPursuit

# sample matrices A and b
A = np.array([[0.1817, 0.5394, -0.1197,  0.6404],
             [0.6198, 0.1994, 0.0946, -0.3121],
             [-0.7634, -0.8181, 0.9883, 0.7018]])
b = np.array([[1.1862], [-0.1158], [-0.1093]])

greedy_algo = GreedyAlgorithm(show_calc=True)
x = greedy_algo.omp(A, b, 2)
"""
Below are expected output
Residual rk
1st iteration: [[0.72570314], [0.10862391], [-0.61394818]]
2nd iteration: [[1.65905921e-06], [1.29754484e-05], [4.25644903e-06]]

L2 Norm of rk
1st iteration: 0.9567531400438528
2nd iteration: 1.3756165787494833e-05
"""

print("Non-zero elements:", np.nonzero(x)[0])
# 1 and 3

# scikit-learn
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=2)
x_sk = omp.fit(A, b).coef_.reshape(-1, 1)

# This number should be very small.
print("square sum of difference between my omp and scikit-learn: {}".format(np.sum((x - x_sk)**2)))
