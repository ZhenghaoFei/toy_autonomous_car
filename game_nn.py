import numpy as np
from matrix_simulator import *

def policy_forward(env, model):
    env = env.ravel()
    env = env.reshape(-1,1)
    h = np.dot(model['W1'], env) #should be 200 * 1
    h[h<0] = 0 # ReLU nonlinearity
    action_score = np.dot(h.T, model['W2']) # should be (1 , 4)
#     print action_score
    action = np.argmax(action_score)
    return action, h, env # return action, and hidden state

# reward should be a vector e.g [0, 0, 1, 0]
def policy_backward(h_cache, env_cache, feedback, model):
    """ backward pass. (h_cache is array of intermediate hidden states) """
    feedback = feedback.reshape(1,-1)
    dW2 = np.dot(h_cache, feedback) # 200 * 4
#     dh = np.outer(reward, model['W2'])
    dh = np.dot(model['W2'], feedback.T) # 200 * 1
    dh[h_cache <= 0] = 0 # backpro prelu
    #epx ex input (D*D, 1)
    dW1 = np.dot(dh, env_cache.T)
    return {'W1':dW1, 'W2':dW2}

# perform sgd parameter update 
def sgd_update(model, gradient, learning_rate):
    for k,v in model.iteritems():
        model[k] += learning_rate * gradient[k]

def save_model(model):
    np.savetxt("w1.txt", model['W1'])
    np.savetxt("w2.txt", model['W2'])

def load_model():
    model = {}
    model['W1'] = np.loadtxt("w1.txt")
    model['W2'] = np.loadtxt("w2.txt")
    return model

# training process
def train_game_nn(model, map_matrix, initial_car_location, learning_rate, max_iter = 10):
    car_location = initial_car_location
    car_location_save =[]
    car_location, feedback, env = simulator(map_matrix, initial_car_location) # initial env
    for i in range(max_iter):
        action, h_cache, env_cache =policy_forward(env, model)
        car_location, feedback, env = simulator(map_matrix, initial_car_location, car_location, action)
        car_location_save.append(car_location)
        gradient = policy_backward(h_cache, env_cache, feedback, model)
        sgd_update(model, gradient, learning_rate)
    return car_location_save
