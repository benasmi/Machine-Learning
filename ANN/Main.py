import numpy as np
from ANNClasifier import ANNClasifier



p = ANNClasifier([3,2,1])

b, w = p.backprop([[ 1],[ 4], [1]], [2])
print(b)
print("W")
print(w)



