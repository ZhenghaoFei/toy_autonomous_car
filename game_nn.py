import numpy as np
from matrix_simulator import *
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
import matplotlib.pyplot as plt
debug = False
def discount_rewards(r):
    gamma = 0.9# discount factor for reward

    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=float)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
    Returns a tuple of:1
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    # loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return probs, dx

def policy_forward(env, model):
    h1 = np.dot(model['W1'], env) #should be 200 * 1
    h1[h1<0] = 0 # ReLU nonlinearity

    h2 = np.dot(h1.T,model['W2'])
    h2[h2<0] = 0
    action_score = np.dot(h2.T, model['W3']) # should be (1 , 4)
    if debug:
        print "h1:", h1.shape
        print "w2:", model['W2'].shape
        print "h2:", h2.shape
        # print action_score
        print action_score.shape
    probs = np.exp(action_score - np.max(action_score))
    # probs = np.exp(action_score)
    probs /= np.sum(probs)
    np.random.seed()

    dice = np.random.uniform() # roll the dice!
    # print dice
    # print "probs:"
    # action = 0
    # for i in range(probs.shape[0]):
    #     prob = np.sum(probs[:i+1])
    #     if dice < prob:
    #         action = i
    #         # print action
    #         break  # if dice fall in certain range chose the action
    # print probs
    action = np.random.choice(4, 1, p = probs)
    # print "action ", action


    N = action_score.shape[0]
    dx = probs.copy()
    dx[action] -= 1 # fake label
    dx /= N
    dx = -dx # grad that encourages the action that was taken to be taken if feedback > 0 
    # print "dx"
    # print dx
    return action, h1, h2, dx # return action, and hidden state

# reward should be a vector e.g [0, 0, 1, 0]
def policy_backward(feedback, h1_cache, h2_cache, env_cache, model, reg):
    """ backward pass. (h_cache is array of intermediate hidden states) """
    if debug:
        print 'feedback:', feedback.shape
        print 'h1_cache:', h1_cache.shape
        print 'h2_cache:', h2_cache.shape
        print 'env_cache:', env_cache.shape
    dW3 = np.dot(h2_cache.T, feedback) # 200 * 4
    dh2 = np.dot(model['W3'], feedback.T) # 200 * 1
    if debug:
        print "dW3:", dW3.shape
        print "dh2:", dh2.shape
    dW2 = np.dot(h1_cache.T, dh2.T) # 200*200
    dh1 = np.dot(dh2.T, model['W2'].T)
    dW1 = np.dot(dh1.T,env_cache)
    if debug:
        print "dh1:", dh1.shape
        print "dW2:", dW2.shape
        print "dW1:", dW1.shape
    # dh1[h1_cache <= 0] = 0 # backpro prelu
    # dh2[h2_cache <= 0] = 0

    dW1 += 0.5 * reg * 2 * model['W1']
    dW2 += 0.5 * reg * 2 * model['W2']
    dW3 += 0.5 * reg * 2 * model['W3']

    return {'W1':dW1, 'W2':dW2, 'W3':dW3}


def sgd_update(model, gradient, learning_rate):
    for k,v in model.iteritems():
        model[k] += learning_rate * gradient[k]

def rmsprop_update(model, gradient, learning_rate, rmsprop_cache, decay_rate = 0.99):
    for k,v in model.iteritems():
        g = gradient[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
    return rmsprop_cache,model

def creat_model(D, H1, H2, C=4):
    model = {}
    model['W1'] = np.random.randn(H1, D) / np.sqrt(D * H1) # "Xavier" initialization
    # model['b1'] = np.random.randn(H1)
    # model['W2'] = np.random.randn(H1,H2) / np.sqrt(H1)
    # model['b2'] = np.random.randn(H2)
    model['W2'] = np.random.randn(H1, H2) / np.sqrt(H1 * H2)
    model['W3'] = np.random.randn(H2, C) / np.sqrt(H2 * C) 
    # model['b2'] = np.random.randn(4)
    if debug:
        print "W1:", model['W1'].shape
        print "W2:", model['W2'].shape
        print "W3:", model['W3'].shape

    return model


def save_model(model):
    for k,v in model.iteritems():
        np.savetxt('%s.txt' %k, model[k])


def load_model(model):
    for k,v in model.iteritems():
        model[k] = np.loadtxt('%s.txt' %k)    
    return model

# training process
def train_game_rlnn(model, map_prameters, learning_rate, reg=0, decay = 0.995, max_iter = 10,plotmap=False):
    dim1, dim2, probobility = map_prameters
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory
    map_matrix, initial_car_location, goal_location = random_map(dim1, dim2, probobility)
    if plotmap:
        plot_map(map_matrix, initial_car_location)
        plt.show()
        plotmap = False
    goal_distance = 10000
    step = 0
    collision = 0
    normal_reset = 0
    simu_round = 0
    arrive = 0
    car_location, feedback, env = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step) # initial env
    # plt.imshow(env, interpolation='none')
    # plt.show()
    # print env
    env = env.ravel()
    # book keeping for one round and initialize
    feedback_round = [] 
    env_round = []
    action_round = []
    h1_round = []
    h2_round = []
    dscore_round = []
    env_round.append(env) # first env

    for i in range(max_iter):
        action, h1, h2, dscore = policy_forward(env, model)
        car_location, feedback, env, goal_distance, step, reset, status = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step, car_location=car_location, action=action)

        if reset:  # one simu_round finished
            # print status
            simu_round += 1
            action_round.append(action)
            dscore_round.append(dscore)
            feedback_round.append(feedback)
            h1_round.append(h1)
            h2_round.append(h2)

            epr = np.vstack(feedback_round)

            # pre process book keeping
            dr = discount_rewards(epr) # discouted rewards for this round, shape: step
            dscore_input = np.vstack(dscore_round)
            feedback_input = np.zeros(len(dr)) # shape: step, 
            for i in range(len(dr)): # transfer reward vector to reward matrix
                feedback_input[i] = dr[i]
            feedback_input = feedback_input.T
            env_input = np.vstack(env_round) # shape: step, dim1*dim2
            h1_input = np.vstack(h1_round)
            h2_input = np.vstack(h2_round)
            feedback_round = [] 
            env_round = []
            action_round =[]
            h1_round = []
            h2_round = []
            dscore_round = []

            
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

            # print feedback_input
            # print status
            # print "feedback: ", feedback_input.shape
            # print "dscore: ", dscore_input.shape
            dscore_input = dscore_input.T
            # print "dscore: "
            # print dscore_input
            # print "feedback_input"
            # print feedback_input
            dscore_input *= feedback_input
            dscore_input = dscore_input.T
            # print "dscore_input: "
            # print dscore_input
            gradient = policy_backward(dscore_input, h1_input, h2_input, env_input, model, reg)
            # sgd_update(model, gradient, learning_rate)
            rmsprop_cache,model = rmsprop_update(model, gradient, learning_rate, rmsprop_cache)

             # initial env
            car_location, feedback, env = simulator(map_matrix, initial_car_location, goal_location, goal_distance, step)
            # print env
            env = env.ravel()
            env_round.append(env) # first env

        # book keeping
        if not reset:
            action_round.append(action)
            dscore_round.append(dscore)
            feedback_round.append(feedback)
            h1_round.append(h1)
            h2_round.append(h2)
            # print env
            env = env.ravel()
            env_round.append(env)

