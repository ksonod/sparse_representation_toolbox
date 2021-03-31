# section 2

import numpy as np
from sparse_toolbox.tools import MutualCoherence

A =np.array([[16,-2,15,13],[5,6,8,8],[9,4,11,12],[4,12,10,1]])

print(A)

mutual_coherence = MutualCoherence(matrix=A)
mu = mutual_coherence.compute_mutual_coherence()

print("Mutual Coherence:{}".format(mu))  # 0.9543
