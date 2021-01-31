# weak matching pursuit
# Michael Elad, Sparse and Redundant Representations, Chapter 3, page 39-41, 2010

import numpy as np

# sample matrices A and b
A = np.array([[0.1817, 0.5394, -0.1197,  0.6404],
             [0.6198, 0.1994, 0.0946, -0.3121],
             [-0.7634, -0.8181, 0.9883, 0.7018]])

b = np.array([[1.1862], [-0.1158], [-0.1093]])

# initialization
t = 0.5  # parameter. The larger it is, the faster (and the less accurate) the algorithm becomes.
x_0 = np.zeros((A.shape[1], 1))  # 4 x 1 zeros. Initial solution
xkm1 = x_0
r_0 = b - np.matmul(A, x_0)  # 3 x 1. Initial residual
column_idx = []
r_km1 = r_0  # r_{k-1}
k = 1
num_iteration = 2  # Number of iterations

if num_iteration > A.shape[1]:
    print("num_iteration cannot exceed {}".format(A.shape[1]))
    num_iteration = A.shape[1]

# Start iteration
for k in range(1, num_iteration+1):
    minimum_Ei = 100000000  # initialization

    print("--iteration {}--".format(k))

    for j in range(A.shape[1]):  # sweep over all the columns of A.

        pi = np.abs(np.dot(A[:, j], r_km1))

        if pi >= t * np.linalg.norm(r_km1):
            column_idx.append(j)
            break

    xk = xkm1
    xk[column_idx, 0] = xk[column_idx, 0] + np.matmul(A[:, column_idx].T, r_km1).flatten()

    print("Column {} is chosen".format(column_idx[-1] + 1))
    # First iteration: 2nd column
    # Second iteration: 4th column

    # residual vector
    r_k = b - np.matmul(A, xk)  # b - A x

    print("Residual:\n{}".format(r_k))
    # First iteration: [[ 0.80529509], [-0.25660912], [ 0.46841284]]
    # Second iteration: [[ 0.21322556], [ 0.03193695], [-0.18042288]]

    print("L2 norm of r_k = {}".format(np.linalg.norm(r_k)))
    # First iteration: 0.9663120678925727
    # Second iteration: 0.281136129633003

    # update for next
    r_km1 = r_k
    xkm1 = xk
