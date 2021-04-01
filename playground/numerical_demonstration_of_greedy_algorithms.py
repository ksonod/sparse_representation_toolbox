import numpy as np
from algorithm.pursuit import GreedyAlgorithm
import matplotlib.pyplot as plt

# Set the maximum number of non-zeros in the generated vector
s_max = 15

# Set the minimum and maximum entry values
min_coeff_val = 1
max_coeff_val = 3

# Number of realizations
num_realizations = 200  # 200

# rendom seed
base_seed = 1

#  Create the dictionary
# Create a random matrix A of size (n x m)
n = 50
m = 100
A = np.random.randn(n, m)
A_normalized = A / np.linalg.norm(A, axis=0)  # normalization

# Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4
# Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4

# Initialization
L2_error = np.zeros((s_max, num_realizations, 2))
support_error = np.zeros((s_max, num_realizations, 2))

greedy_algorithm = GreedyAlgorithm()

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

        # OMP
        x_omp = greedy_algorithm.omp(A_normalized, b, s)
        x_omp[np.abs(x_omp) < eps_coeff] = 0

        # Compute the relative L2 error
        L2_error[s-1, experiment, 0] = np.linalg.norm(x_omp - x) ** 2 / np.linalg.norm(x) ** 2

        # Get the indices of the estimated support
        estimated_supp = np.nonzero(np.abs(x_omp) > eps_coeff)[0]

        # Compute the support recovery error
        # (max{|S_pred|, |S_correct|} - |S_pred cap S_correct|) / max{|S_correct|, |S_pred|}
        support_error[s-1, experiment, 0] = 1 - len(set(true_supp).intersection(set(estimated_supp))) / np.max(
            [len(true_supp), len(estimated_supp)])

        # Basis pursuit via linear programming
        x_lp = greedy_algorithm.basis_pursuit_lp(A_normalized, b, tol_lp)
        x_lp[np.abs(x_lp) < eps_coeff] = 0

        # Compute the relative L2 error
        L2_error[s-1, experiment, 1] = np.linalg.norm(x_lp - x) ** 2 / np.linalg.norm(x) ** 2

        # Get the indices of the estimated support
        estimated_supp = np.nonzero(np.abs(x_lp) > eps_coeff)[0]

        # Compute the support recovery score
        support_error[s-1, experiment, 1] = 1 - len(set(true_supp).intersection(set(estimated_supp))) / np.max(
            [len(true_supp), len(estimated_supp)])

# Display the results
plt.rcParams.update({'font.size': 14})
# Plot the average relative L2 error, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, 0], axis=1), color='red')
plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, 1], axis=1), color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Average and Relative L2-Error')
plt.axis((0, s_max, 0, 1))
plt.legend(['OMP', 'BP by LP'])
plt.show()

# Plot the average support recovery score, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, 0], axis=1), color='red')
plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, 1], axis=1), color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Probability of Error in Support')
plt.axis((0, s_max, 0, 1))
plt.legend(['OMP', 'BP by LP'])
plt.show()