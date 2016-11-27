import numpy as np
import matplotlib.pyplot as plt
reward = np.loadtxt('./reward_tr.txt')
plt.plot(reward)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.show()
