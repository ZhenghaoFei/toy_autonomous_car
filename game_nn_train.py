import numpy as np
import matplotlib.pyplot as plt
from game_nn import *
from matrix_simulator import *

# hyperparameters
H1 = 50 # number of hidden layer neurons
H2 = 50
batch_size = 1 # every how many episodes to do a param update?
lr_decay = 0.995 # decay factor for RMSProp leaky sum of grad^2
learning_rate = 1e-5

resume = False # resume from previous checkpoint?

dim = 10
probobility = 0.1
map_prameters = dim, dim ,probobility
# model initialization
D = (dim + 2) * (dim + 2) # input dimensionality, because 1 pad
model = creat_model(D, H1, H2)
if resume:
    print "model resumed"
    model = load_model(model)


# train
train_game_rlnn(model, map_prameters, learning_rate, reg=1, decay=lr_decay, max_iter=100000)
save_model(model)

# # test
# print "start testing"
# map_matrix, initial_car_location, goal_location = random_map(dim, dim, probobility)
# step = 0
# goal_distance = 10000
# car_location_save =[]
# # initial env
# car_location, feedback, env = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step) 
# for i in range(10):
#     action, h1, h2, dscore = policy_forward(env, model)
#     last_car_location = car_location # keep last car_location
#     car_location, feedback, env, goal_distance, step, reset, status = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step, car_location=car_location, action=action)
#     car_location_save.append(car_location)

#     if reset:  # one simu_round finished
#     	print("game end")
#     	print(status)
#     	break

# # plot
# fig, ax = plt.subplots()  
# print len(car_location_save)
# def animate(i):
#     map_plot = np.copy(map_matrix)
#     map_plot[car_location_save[i]] = 3
#     ax.imshow(map_plot, interpolation='none')
# anim = animation.FuncAnimation(fig, animate, frames = len(car_location_save) )
# plt.show()