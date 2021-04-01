import numpy as np
from scipy.optimize import linprog

class GreedyAlgorithm:
    def __init__(self, show_calc=False):
        self.show_calc = show_calc  # show intermediate calculation process

    def omp(self, A, b, k):
        """
        omp solves the P0 problem via OMP
        min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k

        A: input matrix. numpy array
        b: input vector. numpy array
        k: number of non-zero elements. integer

        See also scikit-learn: https://scikit-learn.org/stable/modules/linear_model.html#omp
        """
        # Normalization
        A = A / np.linalg.norm(A, axis=0)

        # Initialization
        x = np.zeros((A.shape[1], 1))
        rk = b - np.matmul(A, x)
        column_idx = []

        for i in range(k):
            idx = np.argmax(np.abs(np.matmul(A.T, rk)))  # find a column with the maximum inner product.
            column_idx.append(idx)
            As = A[:, column_idx]
            xk = np.matmul(np.linalg.inv(np.matmul(As.T, As)), As.T)
            x[column_idx] = np.matmul(xk, b).reshape(-1, 1)
            rk = b - np.matmul(A, x)

            if self.show_calc:
                print("-Iteration {}---".format(i+1))
                print("Residual:\n{}".format(rk))
                print("L2 norm of r_k = {}\n".format(np.linalg.norm(rk)))
        return x


    def basis_pursuit_lp(self, A, b, tol):
        """
        Solving Basis Pursuit via linear programing
        min_x || x ||_1 s.t. b = Ax

        A: input matrix. numpy array
        b: input vector. numpy array
        tol: tolerance. small positive number
        """

        # Normalization
        A = A / np.linalg.norm(A, axis=0)

        # Set the options to be used by the linprog solver
        opt = {"tol": tol, "disp": False}

        m = A.shape[1]
        obj = np.ones(2 * m)
        Ap = np.concatenate([A, -A], axis=1)
        bound = []

        for i in range(2 * m):
            bound.append((0, float("inf")))  # 0 to inf
        bound = np.array(bound)

        opt = linprog(c=obj,
                      A_ub=None,
                      b_ub=None,
                      A_eq=Ap,
                      b_eq=b,
                      bounds=bound,
                      method='interior-point',
                      callback=None,
                      options=opt,
                      x0=None)

        uv = opt.x.reshape(2, m).T
        x = uv[:, 0] - uv[:, 1]
        return x.reshape(-1, 1)
