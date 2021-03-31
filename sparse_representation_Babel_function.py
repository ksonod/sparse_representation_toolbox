# Section 2
import numpy as np
from sparse_toolbox.tools import babel_function

A = np.array([[16,-2,15,13],[5,6,8,8],[9,4,11,12],[4,12,10,1]])

m1_babel = babel_function(A)
print("Babel:")
print(m1_babel)
# 0.95429585 1.9066768  2.4374598

