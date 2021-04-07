import numpy as np
from scipy.optimize import linprog
from enum import Enum

class PursuitAlgorithmType(Enum):
    omp = 0  # orthogonal matching pursuit
    thresholding = 1  # thresholding
    basis_pursuit_lp = 2  # basis pursuit with linear programming
    ls_omp = 3  # least-square orthogonal matching pursuit

class PursuitAlgorithm:
    def __init__(self, show_calc=False, pursuit_algorithm=PursuitAlgorithmType.omp):
        self.show_calc = show_calc  # show intermediate calculation process
        self.pursuit_algorithm = pursuit_algorithm

    def __call__(self, A, b, t):
        if self.pursuit_algorithm == PursuitAlgorithmType.omp:
            x = self.omp(A, b, t)

        elif self.pursuit_algorithm == PursuitAlgorithmType.thresholding:
            x = self.thresholding(A, b, t)

        elif self.pursuit_algorithm == PursuitAlgorithmType.basis_pursuit_lp:
            x = self.basis_pursuit_lp(A, b, t)

        elif self.pursuit_algorithm == PursuitAlgorithmType.ls_omp:
            x = self.ls_omp(A, b, t)

        return x

    def omp(self, A, b, k):
        """
        Solving the P0 problem via OMP. Concretely,
        min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k,
        where k is integer,
        or
        min_x ||x||_0 s.t. ||y - Ax||_2^2 <= tol,
        where tol is small non-integer.


        A: input matrix. numpy array. Each column is normalized.
        b: input vector. numpy array
        k: number of non-zero elements. integer

        See also scikit-learn: https://scikit-learn.org/stable/modules/linear_model.html#omp
        """
        # Initialization
        x = np.zeros((A.shape[1], 1))
        r = np.copy(b)
        column_idx = []

        # k is the number of non-zero coefficients
        if isinstance(k, int):
            for i in range(k):
                idx = np.argmax(np.abs(np.matmul(A.T, r)))  # find a column with the maximum inner product.
                column_idx.append(idx)
                column_idx.sort()
                As = A[:, column_idx]
                r = b - As @ np.linalg.pinv(As) @ b

                if self.show_calc:
                    print("-Iteration {}---".format(i+1))
                    print("Residual:\n{}".format(r))
                    print("L2 norm of r_k = {}\n".format(np.linalg.norm(r)))

        # k is error tolerance
        else:
            while np.dot(r.T, r) > k:
                idx = np.argmax(np.abs(np.matmul(A.T, r)))  # find a column with the maximum inner product.
                column_idx.append(idx)
                column_idx.sort()
                As = A[:, column_idx]
                r = b - As @ np.linalg.pinv(As) @ b

        x[column_idx] = np.linalg.pinv(As) @ b
        return x

    def basis_pursuit_lp(self, A, b, tol):
        """
        Solving Basis Pursuit via linear programing
        min_x || x ||_1 s.t. b = Ax

        A: input matrix. numpy array. Each column is normalized.
        b: input vector. numpy array
        tol: tolerance. small positive number
        """

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

    def thresholding(self, A, b, thr):
        """
        Threshodling algorithm.
        """
        column_idx = np.argsort(-np.abs(np.matmul(A.T, b)), axis=0).flatten()  # sort in descending order.

        # initialization
        x = np.zeros((A.shape[1], 1))
        r = np.copy(b)
        k = 1

        while np.dot(r.T, r) > thr:
            As = A[:, column_idx[:k]]
            xk = np.matmul(np.linalg.pinv(As), b)
            r = b - np.matmul(As, xk)
            k = k + 1

        x[column_idx[:k-1]] = xk

        return x

    def ls_omp(self, A, b, tol):
        """
        Least-square orthogonal matching pursuit
        """

        m = A.shape[1]  # number of columns

        # initialization
        x = np.zeros((m, 1))
        r = np.copy(b)
        column_idx = []

        while np.dot(r.T, r) > tol:
            residual_inner_prod = np.zeros((m, 1))  # initialization

            for i in range(m):
                column_idx_temp = column_idx + [i]
                r_temp = b - A[:, column_idx_temp] @ np.linalg.pinv(A[:, column_idx_temp]) @ b
                residual_inner_prod[i] = np.dot(r_temp.T, r_temp)

            column_idx.append(np.argmin(residual_inner_prod))
            column_idx.sort()

            r = b - A[:, column_idx] @ np.linalg.pinv(A[:, column_idx]) @ b  # residual

        x[column_idx] = np.linalg.pinv(A[:, column_idx]) @ b

        return x
