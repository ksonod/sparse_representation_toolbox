# Orthogonal matching pursuit
# Michael Elad, Sparse and Redundant Representations, Chapter 3, page 36-37, 2010

import numpy as np
from algorithm.pursuit import PursuitAlgorithmType, PursuitAlgorithm
from sklearn.linear_model import OrthogonalMatchingPursuit

# sample matrices A and b
A = np.array([[0.1817, 0.5394, -0.1197,  0.6404],
             [0.6198, 0.1994, 0.0946, -0.3121],
             [-0.7634, -0.8181, 0.9883, 0.7018]])
b = np.array([[1.1862], [-0.1158], [-0.1093]])

# input data
dat = {
    "A": A,
    "b": b,
    "num_support": 2
}

greedy_algo = PursuitAlgorithm(show_calc=True, pursuit_algorithm=PursuitAlgorithmType.omp)
x = greedy_algo.omp(dat)


"""
Below are expected output
Residual rk
1st iteration: [[0.72572239], [0.10861452], [-0.61392708]]
2nd iteration: [[1.65905921e-06], [1.29754484e-05], [4.25644903e-06]]

L2 Norm of rk
1st iteration: 0.9567531400438529
2nd iteration: 1.3756165787529186e-05
"""

print("Non-zero elements:", np.nonzero(x)[0])
# 1 and 3

# scikit-learn
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=2)
x_sk = omp.fit(A, b).coef_.reshape(-1, 1)

# This number should be very small.  1.2725320679923535e-10
print("square sum of difference between my omp and scikit-learn: {}".format(np.sum((x - x_sk)**2)))
