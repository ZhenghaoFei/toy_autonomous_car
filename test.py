import numpy as np

a = [1,1, 2]
c= [1,2,3]
# d = np.asarray(c)
# print d.reshape(1,-1)
b = np.random.rand(3,4)
print b

N = 3

b[np.arange(N), a] -= 1 # fake label

print b