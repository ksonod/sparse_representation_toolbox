"""
M. Elad, "Sparse and Redundant Representations: From Theory to Applications in Signal and Image Processing,"
 Springer, p230 - 236, (2010)
 Fig. 12.4

 Only MOD is implemented.
"""

import numpy as np
from algorithm.pursuit import PursuitAlgorithmType, PursuitAlgorithm
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

# parameters
rand_seed = 5  # Random seed
num_lin_combination = 4  # number of non-zero elements
num_signals = 4000  # 4000
noise_sigma = 0.1  # standard deviation for a zero-mean Gaussian noise.
eps_coeff = 1e-4  # for OMP
num_iter = 50  # 50
np.random.seed(rand_seed)

# Create a random matrix A of size (n x m): dictionary
n = 30  # dimension  30
m = 60  # number of atoms in a dictionary  60
A_true = np.random.randn(n, m)  # dictionary
A_true = A_true / np.linalg.norm(A_true, axis=0)  # normalization

# Prepare true coefficients
X_true = np.zeros((m, num_signals))  # Coefficient matrix (m x num_signals)
for i in range(num_signals):
    idx = np.sort(np.random.permutation(m)[:num_lin_combination])
    coeff = np.random.randn(num_lin_combination,)
    X_true[idx, i] = coeff

# Initialization of coefficient matrix
X = np.zeros_like(X_true)

# Prepare signal matrix
Y = np.matmul(A_true, X_true)  # n x num_signals
Y = Y + np.random.normal(loc=0, scale=noise_sigma, size=Y.shape)  # Add noise

# Initialization of a dictionary
A = np.copy(Y[:, :60])  # Initialization of dictionary with signal elements Y
A = A / np.linalg.norm(A, axis=0)  # Normalization

mat = loadmat("/Users/kotarosonoda/Downloads/Matlab-Package-Book/matlab.mat")
A_true = mat["Dictionary"]
X_true = mat["coefs"]
Y = mat["data"]
A = np.copy(Y[:, :60])  # Initialization of dictionary with signal elements Y
A = A / np.linalg.norm(A, axis=0)  # Normalization

print("SNR:", np.sum(np.matmul(A_true, X_true).T @ np.matmul(A_true, X_true))/np.sum((Y-np.matmul(A_true, X_true)).T @ (Y-np.matmul(A_true, X_true))))



def restored_elements(D_true, D_comp):
    """
    Computing distances between two dictionaries.
    Input parameters:
    - D_true: true dictionary. numpy array.
    - D_comp: computed dictionary for comparison. numpy array.
    Output:
    - ratio: numpy array.
    """
    thr = 0.01
    counter = 0
    distances = np.abs(D_true.T @ D_comp)

    for i in range(D_true.shape[1]):
        minvalue = 1 - max(distances[i, :])
        counter = counter + (minvalue < thr)

    return 100*counter/D_true.shape[1]  # return ratio

# Initialization
total_err = []
rest_elements = []

# Start iteration
for i in range(num_iter):
    # Pursuit
    dat = {}  # Initialization for pursuit
    for col in range(num_signals):
        dat["A"] = A
        dat["b"] = Y[:, col].reshape(-1, 1)
        dat["num_support"] = num_lin_combination

        greedy_algo = PursuitAlgorithm(pursuit_algorithm=PursuitAlgorithmType.omp)
        x = greedy_algo(dat)
        x[np.abs(x) < eps_coeff] = 0
        X[:, col] = x.flatten()

    # MOD Dictionary update
    A = Y @ X.T @ np.linalg.inv(X @ X.T)
    A = A / np.linalg.norm(A, axis=0)  # normalization

    frobenius_norm = np.linalg.norm(Y - A @ X, ord="fro")**2  # Frobenius norm
    total_err.append(np.sqrt(frobenius_norm/Y.size))  # error

    rest_elements.append(restored_elements(A_true, A))

    print("Iteration:{} | Error: {}".format(i, total_err[i]))

# Data visualization
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(rest_elements, "ko-")
plt.xlabel("Iteration")
plt.ylabel("Relative number of restored atoms")
plt.xlim([0, num_iter-1])
plt.ylim([0, 100])

plt.subplot(122)
plt.plot(total_err, "ko-")
plt.xlabel("Iteration")
plt.ylabel("Average representation error")
plt.xlim([0, num_iter-1])
plt.ylim([0.08, 0.22])
plt.show()
