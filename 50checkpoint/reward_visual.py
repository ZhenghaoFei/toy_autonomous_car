import numpy as np
import matplotlib.pyplot as plt

R = np.loadtxt("reward_tr.txt")

print R.shape
num = R.shape[0]
X = 1000*(np.arange(num)+1)

plt.plot(X, R, linestyle='-', marker='o')
plt.show()
