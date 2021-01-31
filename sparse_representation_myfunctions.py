import numpy as np

def mutual_coherence(A):
    '''
    Calculating mutual coherence
    A: matrix

    returns mu (mutual coherence)
    '''

    h, w = A.shape
    mu_max = 0

    for i in range(w):
        for j in range(i + 1, w):
            mu = np.dot(A[:, i], A[:, j])/np.sqrt(np.dot(A[:, i], A[:, i]))/np.sqrt(np.dot(A[:, j], A[:, j]))

            if mu > mu_max:
                mu_max = mu
    return mu_max


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