import numpy as np
from matrix_simulator import *
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def discount_rewards(r):
    gamma = 0.1 # discount factor for reward

    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=float)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(env, model):
    h = np.dot(model['W1'], env) #should be 200 * 1
    # print "h:", h.shape
    # print "w2:", model['W2'].shape
    h[h<0] = 0 # ReLU nonlinearity
    action_score = np.dot(h.T, model['W2']) # should be (1 , 4)
#     print action_score
    # print action_score.shape
    action = np.argmax(action_score)
    # print action
    return action, h # return action, and hidden state

# reward should be a vector e.g [0, 0, 1, 0]
def policy_backward(feedback, h_cache, env_cache, model):
    """ backward pass. (h_cache is array of intermediate hidden states) """
    # print 'feedback:', feedback.shape
    # print 'h_cache:', h_cache.shape
    dW2 = np.dot(h_cache.T, feedback) # 200 * 4
#     dh = np.outer(reward, model['W2'])
    dh = np.dot(feedback, model['W2'].T) # 200 * 1
    # print 'dh:', dh.shape

    dh[h_cache <= 0] = 0 # backpro prelu
    #epx ex input (D*D, 1)
    # print 'env:', env_cache.shape
    dW1 = np.dot(dh.T, env_cache)
    return {'W1':dW1, 'W2':dW2}

def sgd_update(model, gradient, learning_rate):
    for k,v in model.iteritems():
        model[k] += learning_rate * gradient[k]

def creat_model(D, H1):
    model = {}
    model['W1'] = np.random.randn(H1, D) / np.sqrt(D) # "Xavier" initialization
    # model['b1'] = np.random.randn(H1)
    # model['W2'] = np.random.randn(H1,H2) / np.sqrt(H1)
    # model['b2'] = np.random.randn(H2)
    model['W2'] = np.random.randn(H1,4) / np.sqrt(H1)
    # model['b2'] = np.random.randn(4)
    return model


def save_model(model):
    for k,v in model.iteritems():
        np.savetxt('%s.txt' %k, model[k])


def load_model(model):
    for k,v in model.iteritems():
        model[k] = np.loadtxt('%s.txt' %k)    
    return model

# training process
def train_game_rlnn(model, map_prameters, learning_rate, decay = 0.995, max_iter = 10):
    dim1, dim2, probobility = map_prameters
    map_matrix, initial_car_location, goal_location = random_map(dim1, dim2, probobility)
    goal_distance = 10000
    step = 0
    collision = 0
    normal_reset = 0
    simu_round = 0
    arrive = 0
    car_location, feedback, env = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step) # initial env
    env = env.ravel()
    # book keeping for one round and initialize
    feedback_round = [] 
    env_round = []
    action_round = []
    h_round = [] 
    env_round.append(env) # first env

    for i in range(max_iter):
        action, h = policy_forward(env, model)
        car_location, feedback, env, goal_distance, step, reset, status = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step, car_location=car_location, action=action)
        env = env.ravel()



        if reset:  # one simu_round finished
            simu_round += 1
            action_round.append(action)
            h_round.append(h)
            feedback_round.append(feedback)

            epr = np.vstack(feedback_round)

            # pre process book keeping
            dr = discount_rewards(epr) # discouted rewards for this round, shape: step
            # print epr
            # print status
            feedback_input = np.zeros([len(dr), 4]) # shape: step, 4
            for i in range(len(dr)): # transfer reward vector to reward matrix
                feedback_input[i, action_round[i]] = dr[i]
            env_input = np.vstack(env_round) # shape: step, dim1*dim2
            h_input = np.vstack(h_round)
            feedback_round = [] 
            env_round = []
            action_round =[]
            h_round = []
            
            # benchmark
            if status == 'collision':
                collision += 1
            elif status == 'arrive':
                arrive += 1
            else:
                normal_reset += 1
            if simu_round%100 ==0:
                learning_rate *= decay
                print "lr: ", learning_rate
                print "collision: %i %%" %(collision)
                print "arrive: %i %%" %(arrive)
                print "normal_reset: %i %%" %(normal_reset)
                collision = 0
                arrive = 0
                normal_reset = 0
            map_matrix, initial_car_location, goal_location = random_map(dim1, dim2, probobility)
            step = 0
             # initial env
            car_location, feedback, env = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step)
            env = env.ravel()

            env_round.append(env) # first env

            gradient = policy_backward(feedback_input, h_input, env_input, model)
            sgd_update(model, gradient, learning_rate)

        # book keeping
        if not reset:
            action_round.append(action)
            feedback_round.append(feedback)
            h_round.append(h)
            env_round.append(env)

