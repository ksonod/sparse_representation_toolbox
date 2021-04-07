"""
M. Elad, "Sparse and Redundant Representations: From Theory to Applications in Signal and Image Processing,"
 Springer, p47, (2010)
 Figs. 3.5 and 3.6
"""

import numpy as np
from algorithm.pursuit import GreedyAlgorithm, PursuitAlgorithm
import matplotlib.pyplot as plt

# Set the maximum number of non-zeros in the generated vector
s_max = 10  # 15

# Set the minimum and maximum entry values
min_coeff_val = 1  # 1
max_coeff_val = 2  # 3

# Number of realizations
num_realizations = 200  # 200

# rendom seed
base_seed = 0

#  Create the dictionary
# Create a random matrix A of size (n x m)
n = 30  # 50
m = 50  # 100
A = np.random.randn(n, m)
A_normalized = A / np.linalg.norm(A, axis=0)  # normalization

# Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4
# Set the optimality tolerance of the linear programing solver
tolerance = 1e-4

# Number of algorithms
num_algo = 3

# Initialization
L2_error = np.zeros((s_max, num_realizations, num_algo))
support_error = np.zeros((s_max, num_realizations, num_algo))


# Loop over the sparsity level
for s in range(1, s_max+1):
    print("{}/{}".format(s, s_max))

    # Use the same random seed in order to reproduce the results if needed
    np.random.seed(s + base_seed)

    # Loop over the number of realizations
    for experiment in range(num_realizations):
        # In this part we will generate a test signal b = A_normalized @ x by
        # drawing at random a sparse vector x with s non-zeros entries in
        # true_supp locations with values in the range of [min_coeff_val, max_coeff_val]
        x = np.zeros((m, 1))

        # Random true_supp vector. This determines indices which give non-zero entries in x
        true_supp = np.random.permutation(m)[:s]

        # random non-zero entries
        rand_sign = (np.random.rand(s).reshape(-1, 1) > 0.5) * 2 - 1
        x[true_supp] = rand_sign * ((max_coeff_val - min_coeff_val) * np.random.rand(s).reshape(-1, 1) + min_coeff_val)

        # signal b
        b = np.matmul(A_normalized, x)

        # Start pursuit algorithm.
        for i in range(num_algo):
            greedy_algo = GreedyAlgorithm(pursuit_algorithm=PursuitAlgorithm(i))
            x_pursuit = greedy_algo(A_normalized, b, tolerance)
            x_pursuit[np.abs(x_pursuit) < eps_coeff] = 0

            # Compute the relative L2 error
            L2_error[s - 1, experiment, i] = np.min([np.linalg.norm(x_pursuit - x) ** 2 / np.linalg.norm(x) ** 2, 1])

            # Get the indices of the estimated support
            estimated_supp = np.nonzero(np.abs(x_pursuit) > eps_coeff)[0]

            # Compute the support recovery error
            # (max{|S_pred|, |S_correct|} - |S_pred cap S_correct|) / max{|S_correct|, |S_pred|}
            support_error[s-1, experiment, i] = 1 - len(set(true_supp).intersection(set(estimated_supp))) / np.max([len(true_supp), len(estimated_supp)])

# Display the results
plt.rcParams.update({'font.size': 14})
# Plot the average relative L2 error, obtained by the OMP and BP versus the cardinality
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, 0], axis=1), color='red', linestyle='-', marker='o')
plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, 1], axis=1), color='magenta', linestyle='-', marker='o')
plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, 2], axis=1), color='green', linestyle='-', marker='o')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Average and Relative L2-Error')
plt.xlim([0, s_max])
plt.ylim([-0.01, 1.01])
plt.legend(['OMP', 'Thr', 'BP by LP'], loc='upper left')

# Plot the average support recovery score, obtained by the OMP and BP versus the cardinality
plt.subplot(122)
plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, 0], axis=1), color='red', linestyle='-', marker='o')
plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, 1], axis=1), color='magenta', linestyle='-', marker='o')
plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, 2], axis=1), color='green', linestyle='-', marker='o')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Probability of Error in Support')
plt.xlim([0, s_max])
plt.ylim([-0.01, 1.01])
plt.legend(['OMP', 'Thr','BP by LP'], loc='upper left')
plt.show()

