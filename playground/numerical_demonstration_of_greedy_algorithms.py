"""
M. Elad, "Sparse and Redundant Representations: From Theory to Applications in Signal and Image Processing,"
 Springer, p47, (2010)
 Figs. 3.5, 3.6, 3.8, and 3.9
"""

import numpy as np
from algorithm.pursuit import PursuitAlgorithmType, PursuitAlgorithm
import matplotlib.pyplot as plt


# Create a random matrix A of size (n x m): dictionary
n = 30
m = 50
A = np.random.randn(n, m)
A_normalized = A / np.linalg.norm(A, axis=0)  # normalization

# [-max_coeff_val, -min_coeff_val]OR[min_coeff_val, max_coeff_val]
min_coeff_val = 1  # minimum value of a random true vector
max_coeff_val = 2  # maximum value of a random true vector

num_iter = 200  # Number of iterations
s_max = 10  # Maximum number of non-zero entries in the solution vector
eps_coeff = 1e-4  # If the entries in the estimated vector are smaller than eps_coeff, we will neglect those entries.
tolerance = 1e-4  # Tolerance for convergence
num_algo = 6  # Number of algorithms
base_seed = 1  # Random seed
p = 0.5  # irls Lp norm.

# getting the name of algorithms
algo_name_list = []
for i in range(num_algo):
    algo_name_list.append(PursuitAlgorithmType(i).name)

# color list corresponding to each algorithm
color_list = ["r-o", "m-o", "g-o", "b-o", "c-o", "k-o"]

# Initialization
L2_error = np.zeros((s_max, num_iter, num_algo))
support_error = np.zeros((s_max, num_iter, num_algo))


# Loop over the sparsity level
for s in range(1, s_max+1):
    print("Progress: {}/{}".format(s, s_max))  # Showing progress

    np.random.seed(s + base_seed)  # Random seed

    # Start experiment
    for experiment in range(num_iter):
        x = np.zeros((m, 1))

        # Random true_supp vector. This determines which indices give non-zero entries in x
        true_supp = np.random.permutation(m)[:s]

        # random non-zero entries [-max_coeff_val, -min_coeff_val]OR[min_coeff_val, max_coeff_val]
        rand_sign = (np.random.rand(s).reshape(-1, 1) > 0.5) * 2 - 1
        x[true_supp] = rand_sign * ((max_coeff_val - min_coeff_val) * np.random.rand(s).reshape(-1, 1) + min_coeff_val)

        # signal b
        b = np.matmul(A_normalized, x)

        dat = {
            "A": A_normalized,
            "b": b,
            "tol": tolerance,
            "p": p
        }

        # Pursuit algorithm.
        for i in range(num_algo):
            greedy_algo = PursuitAlgorithm(pursuit_algorithm=PursuitAlgorithmType(i))
            x_pursuit = greedy_algo(dat)
            x_pursuit[np.abs(x_pursuit) < eps_coeff] = 0

            # Relative L2 error
            L2_error[s - 1, experiment, i] = np.min([np.linalg.norm(x_pursuit - x) ** 2 / np.linalg.norm(x) ** 2, 1])

            # Indices of the estimated support
            estimated_supp = np.nonzero(np.abs(x_pursuit) > eps_coeff)[0]

            # Support recovery error
            # (max{|S_pred|, |S_correct|} - |S_pred cap S_correct|) / max{|S_correct|, |S_pred|}
            support_error[s-1, experiment, i] = 1 - len(set(true_supp).intersection(set(estimated_supp))) / np.max([len(true_supp), len(estimated_supp)])

# Data visualization
plt.rcParams.update({'font.size': 14})

# Average relative L2 error vs cardinality
plt.figure(figsize=(14, 5))
plt.subplot(121)
for i in range(num_algo):
    plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, i], axis=1), color_list[i])
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Average and Relative L2-Error')
plt.xlim([0, s_max])
plt.ylim([-0.01, 1.01])
plt.legend(algo_name_list, loc='upper left')

# Average support recovery score vs cardinality
plt.subplot(122)
for i in range(num_algo):
    plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, i], axis=1), color_list[i])
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Probability of Error in Support')
plt.xlim([0, s_max])
plt.ylim([-0.01, 1.01])
plt.legend(algo_name_list, loc='upper left')
plt.show()

