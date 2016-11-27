import numpy as np
from game_nn import *

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 1 # every how many episodes to do a param update?
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

dim1 = 10
dim2 = 10
probobility = 0.3
map_prameters = dim1, dim2 ,probobility
# model initialization
D = (dim1 + 2) * (dim2 + 2) # input dimensionality, because 1 pad
model = creat_model(D, H)

learning_rate = 1e-4
train_game_rlnn(model, map_prameters, learning_rate, max_iter=10)
