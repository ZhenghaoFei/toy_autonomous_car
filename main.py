import numpy as np
import matplotlib.pyplot as plt
from game_nn import *
from matrix_simulator import *

# hyperparameters
H1 = 200 # number of hidden layer neurons
H2 = 100
batch_size = 1 # every how many episodes to do a param update?
lr_decay = 0.9995 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

dim1 = 10
dim2 = 10
probobility = 0.2
map_prameters = dim1, dim2 ,probobility
# model initialization
D = (dim1 + 11) * (dim2 + 11) # input dimensionality, because 1 pad
model = creat_model(D, H1, H2)
if resume:
    print "model resumed"
    model = load_model(model)


# initialize environment
# map_matrix = np.array\
#      ([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  1.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.],
#        [ 0.,  0.,  0.,  1.,  1,  1.,  1.,  1.,  1.,  1.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])



# goal_location = 7, 8 # define goal location
# map_matrix[goal_location] = 2
# initial_car_location = 1, 1 # initial car location x and y
# car_location = initial_car_location
# car_location_save =[]

# train
learning_rate = 1e-5
train_game_rlnn(model, map_prameters, learning_rate, reg=1, decay=lr_decay, max_iter=2000000)
save_model(model)

# # test
# print "start testing"
# map_matrix, initial_car_location, goal_location = random_map(10, 10, 0.3)
# step = 0
# goal_distance = 10000
# car_location_save =[]
# car_location, feedback, env = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step) # initial env
# for i in range(20):
#     action, h_cache, env_cache = policy_forward(env, model)
#     car_location, feedback, env, _, step, _ = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step, car_location=car_location, action=action)
#     car_location_save.append(car_location)
#     if np.any(feedback == 1):
#         print "game end"
#         break
#     if np.any(feedback == -1):
#         print "game end"
#         break

# # plot
# fig, ax = plt.subplots()  
# print len(car_location_save)
# def animate(i):
#     map_plot = np.copy(map_matrix)
#     map_plot[car_location_save[i]] = 3
#     ax.imshow(map_plot, interpolation='none')
# anim = animation.FuncAnimation(fig, animate, frames = len(car_location_save) )
# plt.show()