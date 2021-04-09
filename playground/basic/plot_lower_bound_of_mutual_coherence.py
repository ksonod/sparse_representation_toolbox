"""
Plotting lower bound of mutual coherence as a function of m for different n.
"""

import numpy as np
import matplotlib.pyplot as plt
from sparse_toolbox.tools import MutualCoherence

for n in range(1,10):
    lower_bound = []
    for m in range(n+1,n+11,1):
        mutual_coherence = MutualCoherence()
        lower_bound.append(mutual_coherence.mu_lower_bound(np.ones((n,m))))

    plt.plot(np.linspace(n+1,n+10,10), np.array(lower_bound), 'o-', label="n={}".format(n))
plt.legend()
plt.xlabel('m')
plt.ylabel('Lower bound of mutual coherence')
plt.show()

