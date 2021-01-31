# weak matching pursuit

import numpy as np

# sample matrices A and b
A = np.array([[0.1817, 0.5394, -0.1197,  0.6404],
             [0.6198, 0.1994, 0.0946, -0.3121],
             [-0.7634, -0.8181, 0.9883, 0.7018]])

b = np.array([[1.1862], [-0.1158], [-0.1093]])


num_iteration = 4  # Number of iterations
epsilon = 0.1 # If the L2 norm of the residual vector is smaller than this value, iteration is ended.

if num_iteration > A.shape[1]:
    print("num_iteration cannot exceed {}".format(A.shape[1]))
    num_iteration = A.shape[1]

column_idx = np.argsort(-np.abs(np.matmul(A.T, b)),axis = 0).flatten() # Sort in descending order.

# Start iteration
for k in range(1, num_iteration+1):

    print("\--iteration {}--".format(k))

    xk = np.matmul(A[:,np.sort(column_idx[:k])].T, b)

    print("Column {} is chosen".format(column_idx[k-1]+1))
    # First iteration: 4th column
    # Second iteration: 2nd column
    # Third iteration: 3rd column
    # Fourth iteration: 1st column

    # residual vector
    r_k = b - np.matmul(A[:,np.sort(column_idx[:k])], xk)  # b - A x

    print("Residual:\n{}".format(r_k))
    # First iteration: [[0.72570314], [0.10862391], [-0.61394818]]
    # Second iteration: [[ 0.34479823], [-0.03218521], [-0.03623534]]
    # Third iteration: [[ 0.31356084], [-0.00749802], [ 0.22167539]]
    # Fourth iteration: [[ 0.27227872], [-0.14831616], [ 0.39511935]]

    print("L2 norm of r_k = {}".format(np.linalg.norm(r_k)))
    # First iteration:  0.956753140516202
    # Second iteration: 0.34818774738365016
    # Third iteration: 0.38407889605775153
    # Fourth iteration: 0.5022476346045851

    # update for next
    r_km1 = r_k
    xkm1 = xk

    if np.linalg.norm(r_k) <= epsilon:
        print('Finished...')
        break