# Orthogonal matching pursuit
# Michael Elad, Sparse and Redundant Representations, Chapter 3, page 36-37, 2010

import numpy as np

# sample matrices A and b
A = np.array([[0.1817, 0.5394, -0.1197,  0.6404],
             [0.6198, 0.1994, 0.0946, -0.3121],
             [-0.7634, -0.8181, 0.9883, 0.7018]])

b = np.array([[1.1862], [-0.1158], [-0.1093]])

# initialization
x_0 = np.zeros((A.shape[1], 1))  # 4 x 1 zeros. Initial solution
r_0 = b - np.matmul(A, x_0)  # 3 x 1. Initial residual
r_km1 = r_0  # r_{k-1}
column_idx = []  # 1 x 4
max_inner_product = 0
num_iteration = 2  # Number of iterations

# wrong case
if num_iteration > A.shape[1]:
    print("num_iteration cannot exceed {}".format(A.shape[1]))
    num_iteration = A.shape[1]

# Start iteration
for k in range(1, num_iteration+1):
    max_inner_product = 0  # initialization

    print("--iteration {}--".format(k))

    # Finding minimum error is equivalent to finding the largest inner product between the residual r_{k-1} and
    # the normalized vectors of the matrix A. See page 37.
    for j in range(A.shape[1]):  # sweep over all the columns of A.
        inner_prod = np.dot(A[:, j], r_km1)**2

        if inner_prod > max_inner_product and j not in column_idx:
            max_inner_product = inner_prod
            max_column_idx = j

    column_idx.append(max_column_idx)
    print("Maximum Inner Product is found at the column {}".format(column_idx[-1] + 1))

    As = A[:, column_idx]

    xk = np.matmul(np.linalg.inv(np.matmul(As.T, As)), As.T)  # (As.T As)^-1 As.T b
    xk = np.matmul(xk, b)

    # residual vector
    r_k = b - np.matmul(As, xk)  # b - A x = b - As xk

    print("Residual:\n{}".format(r_k))
    # 1st iteration: [[0.72570314], [0.10862391], [-0.61394818]]
    # 2nd iteration: [[1.65905921e-06], [1.29754484e-05], [4.25644903e-06]]

    print("L2 norm of r_k = {}".format(np.linalg.norm(r_k)))
    # 1st iteration: 0.9567531400438528
    # 2nd iteration: 1.3756165787494833e-05

    r_km1 = r_k  # update for next
