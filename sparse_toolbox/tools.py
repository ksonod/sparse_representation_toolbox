import numpy as np

class MutualCoherence:
    def __init__(self, matrix):
        self.A = matrix
        self.h, self.w = matrix.shape

    def compute_mutual_coherence(self):
        '''
        Calculating mutual coherence
        reference:  M. Elad, Sparse and redundant representations, Eqs. (2.3) and (2.21)

        A: matrix
        returns mu (mutual coherence)
        '''

        mu_max = 0

        for i in range(self.w):
            for j in range(i + 1, self.w):
                mu = np.dot(self.A[:, i], self.A[:, j])/np.sqrt(np.dot(self.A[:, i], self.A[:, i]))/np.sqrt(np.dot(self.A[:, j], self.A[:, j]))

                if mu > mu_max:
                    mu_max = mu
        return mu_max

    def mu_lower_bound(self):
        return np.sqrt((self.w - self.h) / self.h / (self.w - 1))


def babel_function(A):
    A_tilda = np.copy(A)
    A_tilda = A_tilda.astype(float)

    for j in range(A.shape[1]):  # column-wise operation
        l2_norm = np.sqrt(np.dot(A[:, j], A[:, j]))
        A_tilda[:, j] = A[:, j] / l2_norm

    print("Mutual coherence should be the same for A and A_tilda.")
    print("mutual coherence of A: {}".format(mutual_coherence(A)))
    print("mutual coherence of A_tilda: {}".format(mutual_coherence(A_tilda)))

    # Gram matrix
    G = np.dot(A_tilda.T, A_tilda)
    print("\nGram Matrix of A_tilda:")
    print(G)

    # Sorted Gram matrix
    Gs = -np.sort(-G, axis=1)
    print("\nSorted Gram Matrix:")
    print(Gs)

    m1_babel_matrix = np.zeros((Gs.shape[0], Gs.shape[1] - 1))
    for j in range(1, m1_babel_matrix.shape[1] + 1):
        a = np.sum(Gs[:, 1:j + 1], axis=1)
        m1_babel_matrix[:, j - 1] = a

        m1_babel = np.max(m1_babel_matrix, axis=0)
    return m1_babel