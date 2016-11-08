import numpy as np
import matplotlib.pyplot as plt
from game_nn import *

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 1 # every how many episodes to do a param update?
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

# model initialization
D = 7 * 7 # input dimensionality: 80x80 grid
if resume:
	model = load_model()
else:
	model = {}
	model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
	model['W2'] = np.random.randn(H,4) / np.sqrt(H)

# initialize environment
map_matrix = np.array\
     ([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  1.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 0.,  0.,  0.,  1.,  1,  1.,  1.,  1.,  1.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

initial_car_location = 1, 1 # initial car location x and y
car_location = initial_car_location
car_location_save =[]

# train
learning_rate = 1e-4
car_location_save = train_game_nn(model, map_matrix, initial_car_location,  learning_rate,max_iter=100)

# test
# car_location, feedback, env = simulator(map_matrix, initial_car_location) # initial env
# for i in range(20):
#     action, h_cache, env_cache =policy_forward(env, model)
#     car_location, feedback, env = simulator(map_matrix, initial_car_location, car_location, action)
#     car_location_save.append(car_location)
#     if np.any(feedback < 0):
#         print "game end"
#         break
#     if np.any(feedback < 0):
#         print "game end"
#         break

# plot
fig, ax = plt.subplots()  
def animate(i):
    map_plot = np.copy(map_matrix)
    map_plot[car_location_save[i]] = 3
    ax.imshow(map_plot, interpolation='none')

anim = animation.FuncAnimation(fig, animate, frames = len(car_location_save))
plt.show()
