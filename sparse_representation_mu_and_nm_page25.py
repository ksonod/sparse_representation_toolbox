# Section 2
# See equation at the bottom of the page 25

import numpy as np
import matplotlib.pyplot as plt

def mu_lower_bound(m,n):
    return np.sqrt((m-n)/n/(m-1))

for n in range(1,10):
    m = np.linspace(n+1,n+10,10)
    plt.plot(m,mu_lower_bound(m,n), label="n = {}".format(n))
plt.legend()
plt.xlabel('m')
plt.ylabel('Mutual Coherence Limit')

plt.show()

