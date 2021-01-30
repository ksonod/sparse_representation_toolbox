# section 2

import numpy as np
from sparse_representation_myfunctions import mutual_coherence

A =np.array([[16,-2,15,13],[5,6,8,8],[9,4,11,12],[4,12,10,1]])

print(A)

mu = mutual_coherence(A)

print("Mutual Coherence:{}".format(mu)) # 0.9543
