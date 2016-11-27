import numpy as np

a = []
for i in range(5):
	a.append(i)

print a

b = np.vstack(a)
print b
print b.shape