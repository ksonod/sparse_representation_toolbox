# Orthogonal matching pursuit
# Michael Elad, Sparse and Redundant Representations, Chapter 3, page 37-38, 2010

import numpy as np

# sample matrices A and b
A = np.array([[0.1817, 0.5394, -0.1197,  0.6404],
             [0.6198, 0.1994, 0.0946, -0.3121],
             [-0.7634, -0.8181, 0.9883, 0.7018]])

b = np.array([[1.1862], [-0.1158], [-0.1093]])

# initialization
x_0 = np.zeros((A.shape[1], 1))  # 4 x 1 zeros. Initial solution
r_0 = b - np.matmul(A, x_0)  # 3 x 1. Initial residual
column_idx = []
r_km1 = r_0  # r_{k-1}
max_inner_product = 0

# parameter
num_iteration = 2  # Number of iterations

# wrong case
if num_iteration > A.shape[1]:
    print("num_iteration cannot exceed {}".format(A.shape[1]))
    num_iteration = A.shape[1]

# Start iteration
for k in range(1, num_iteration+1):
    minimum_Ei = 100000000  # Initialization. Large arbitrary number.

    print("\n--iteration {}--".format(k))

    # Finding minimum error.
    for j in range(A.shape[1]):  # sweep over all the columns of A.

        if j not in column_idx:
            As = A[:, column_idx + [j]]

            xk = np.matmul(np.linalg.inv(np.matmul(As.T, As)), As.T)  # (As.T As)^-1 As.T b
            xk = np.matmul(xk, b)

            r_k = b - np.matmul(As, xk)  # Residual vector. b - A x = b - As xk
            Ei = np.linalg.norm(r_k)

            if Ei < minimum_Ei:
                min_r_k = r_k
                minimum_Ei = Ei
                min_column_idx = j
                min_xk = xk
                min_As = As

    column_idx.append(min_column_idx)

    print("Minimum error is found at the column {}".format(min_column_idx+1))

    print("Residual:\n{}".format(min_r_k))
    # 1st iteration: [[0.72572239], [0.10861452], [-0.61392708]]
    # 2nd iteration: [[1.65905921e-06], [1.29754484e-05], [4.25644903e-06]]

    print("L2 norm of r_k = {}".format(np.linalg.norm(min_r_k)))
    # 1st iteration: 0.9567531400438528
    # 2nd iteration: 1.375616578750342e-05

    r_km1 = min_r_k  # update for next
