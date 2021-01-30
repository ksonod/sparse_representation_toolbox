# weak matching pursuit

# NOT COMPLETED YET.

import os
import numpy as np

# sample matrices A and b
A = np.array([[0.1817, 0.5394, -0.1197,  0.6404],
            [0.6198, 0.1994, 0.0946, -0.3121],
            [-0.7634, -0.8181, 0.9883, 0.7018]])

b = np.array([[1.1862], [-0.1158], [-0.1093]])

# initialization
x_0 = np.zeros((A.shape[1], 1))  # 4 x 1 zeros. Initial solution
xkm1 = x_0
r_0 = b - np.matmul(A, x_0)  # 3 x 1. Initial residual
column_idx = np.zeros(A.shape[1]).astype("int8")-1  # 1 x 4
As_temp = np.zeros_like(A)

r_km1 = r_0  #r_{k-1}
t = 0.5

max_inner_product = 0
x = np.zeros((4, 1))

k = 1
num_iteration = 2  # Number of iterations

if num_iteration > A.shape[1]:
    print("num_iteration cannot exceed {}".format(A.shape[1]))
    num_iteration = A.shape[1]

# Start iteration
for k in range(1, num_iteration+1):
    minimum_Ei = 100000000  # initialization

    print("--iteration {}--".format(k))

    # Finding minimum error. This part is different from OMP.
    for j in range(A.shape[1]):  # sweep over all the columns of A.

        pi = np.dot(A[:,j],r_km1)

        if pi >= t * np.linalg.norm(r_km1):
            column_idx[k - 1] = j

    xk = xkm1


    As_temp[:, column_idx[k-1]] = A[:, column_idx[k-1]]
    As = As_temp[:, ~np.all(As_temp == 0, axis=0)]

    print("Minimum error is found at {}-th column".format(column_idx[k-1]+1))

    # residual vector
    r_k = b - np.matmul(As, xk)  # b - A x = b - As xk

    print("Residual:\n{}".format(r_k))
    # 1st iteration: [[0.72570314], [0.10862391], [-0.61394818]]
    # 2nd iteration: [[1.65905921e-06], [1.29754484e-05], [4.25644903e-06]]

    print("L2 norm of r_k = {}".format(np.linalg.norm(r_k)))

    r_km1 = r_k # update for next
