import numpy as np
from scipy.optimize import linprog
from enum import Enum

class PursuitAlgorithmType(Enum):
    omp = 0  # orthogonal matching pursuit
    thresholding = 1  # thresholding
    basis_pursuit_lp = 2  # basis pursuit with linear programming
    ls_omp = 3  # least-square orthogonal matching pursuit
    mp = 4  # matching pursuit
    irls = 5  # iterative-reweighted-least-squares (IRLS)

class PursuitAlgorithm:
    def __init__(self, show_calc=False, pursuit_algorithm=PursuitAlgorithmType.omp):
        self.show_calc = show_calc  # show intermediate calculation process
        self.pursuit_algorithm = pursuit_algorithm

    def __call__(self, dat):
        if self.pursuit_algorithm == PursuitAlgorithmType.omp:
            x = self.omp(dat)

        elif self.pursuit_algorithm == PursuitAlgorithmType.thresholding:
            x = self.thresholding(dat)

        elif self.pursuit_algorithm == PursuitAlgorithmType.basis_pursuit_lp:
            x = self.basis_pursuit_lp(dat)

        elif self.pursuit_algorithm == PursuitAlgorithmType.ls_omp:
            x = self.ls_omp(dat)

        elif self.pursuit_algorithm == PursuitAlgorithmType.mp:
            x = self.mp(dat)

        elif self.pursuit_algorithm == PursuitAlgorithmType.irls:
            x = self.irls(dat)

        return x

    def omp(self, dat):
        """
        Solving the P0 problem via OMP. Concretely,
        min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k,
        where k is integer,
        or
        min_x ||x||_0 s.t. ||b - Ax||_2^2 <= tol,
        where tol is small non-integer.


        A: input matrix. numpy array. Each column is normalized.
        b: input vector. numpy array
        k: number of non-zero elements. integer

        See also scikit-learn: https://scikit-learn.org/stable/modules/linear_model.html#omp
        """
        # Initialization
        A = dat["A"]
        b = dat["b"]
        num_support_exist = False

        if "tol" in dat.keys():
            tol = dat["tol"]
        elif "num_support" in dat.keys():
            num_support_exist = True
            k = dat["num_support"]

        x = np.zeros((A.shape[1], 1))
        r = np.copy(b)
        column_idx = []

        # k is the number of non-zero coefficients
        if num_support_exist:
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

        # Tolerance is used.
        else:
            while np.dot(r.T, r) > tol:
                idx = np.argmax(np.abs(np.matmul(A.T, r)))  # find a column with the maximum inner product.
                column_idx.append(idx)
                column_idx.sort()
                As = A[:, column_idx]
                r = b - As @ np.linalg.pinv(As) @ b

        x[column_idx] = np.linalg.pinv(As) @ b
        return x

    def basis_pursuit_lp(self, dat):
        """
        Solving Basis Pursuit via linear programing
        min_x || x ||_1 s.t. b = Ax

        A: input matrix. numpy array. Each column is normalized.
        b: input vector. numpy array
        tol: tolerance. small positive number
        """

        # Initialization
        A = dat["A"]
        b = dat["b"]
        tol = dat["tol"]

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

    def thresholding(self, dat):
        """
        Threshodling algorithm.
        """
        # Initialization
        A = dat["A"]
        b = dat["b"]
        tol = dat["tol"]
        r = np.copy(b)  # residual
        supp = []  # support

        column_idx = np.argsort(-np.abs(np.matmul(A.T, b)).flatten())  # sort in descending order.
        column_idx = column_idx.tolist()  # convert numpy array into list.
        n, m = A.shape

        while np.dot(r.T, r) > tol:
            supp.append(column_idx[len(supp)])
            x = np.zeros((m, 1))
            x[supp] = np.linalg.pinv(A[:, supp].reshape(n, -1)) @ b
            r = b - A[:, supp].reshape(n, -1) @ x[supp].reshape(-1, 1)
        return x

    def ls_omp(self, dat):
        """
        Least-square orthogonal matching pursuit
        """

        # Initialization
        A = dat["A"]
        m = A.shape[1]  # number of columns
        b = dat["b"]
        tol = dat["tol"]
        x = np.zeros((m, 1))
        r = np.copy(b)
        column_idx = []


        # initialization

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

    def mp(self, dat):
        """
        Matching pursuit (mp) algorithm.
        """

        # initialization
        A = dat["A"]
        n, m = A.shape
        b = dat["b"]
        tol = dat["tol"]
        x = np.zeros((m, 1))
        r = np.copy(b)

        while np.dot(r.T, r) > tol:
            idx = np.argmax(np.abs(np.matmul(A.T, r)))
            x[idx] = x[idx] + A[:, idx].T @ r

            As = A[:, idx].reshape(n, -1)
            r = r - As @ As.T @ r
        return x

    def irls(self, dat):
        """
        Iterative-reweighted-least-squares (IRLS) is a relaxation method, where l0 is relaxed to lp and lp is expressed
        as a weighted l2-norm.
        (M_k):  min_x ||X_(k-1)^+ x||_2^2 s.t. b = Ax
        """

        # initialization
        A = dat["A"]
        n, m = A.shape
        b = dat["b"]
        tol = dat["tol"]
        p = dat["p"]
        x = np.ones((m, 1))
        X = np.eye(m)
        x_prev = np.zeros_like(x)
        diff = 1000000  # arbitrary large number

        while diff > tol:
            X2 = X @ X

            # M. Elad, "Sparse and Redundant Representations:
            # From Theory to Applications in Signal and Image Processing,"
            # Springer, p49, Eq. (3.34) (2010)
            x = X2 @ A.T @ np.linalg.pinv(A @ X2 @ A.T) @ b

            np.fill_diagonal(X, np.abs(x)**(1-0.5*p))

            diff = np.linalg.norm(x-x_prev)
            x_prev = x

        return x