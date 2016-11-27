import numpy as np
from matrix_simulator import *
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from simulator_gymstyle import *
from network import *

import matplotlib.pyplot as plt

debug = False


def creat_model(D, H1, H2, C=4):
    model = {}
    model['W1'] = np.random.randn(D, H1) / np.sqrt(D * H1) # "Xavier" initialization
    model['b1'] = 0
    # model['W2'] = np.random.randn(H1,H2) / np.sqrt(H1)
    # model['b2'] = np.random.randn(H2)
    model['W2'] = np.random.randn(H1, H2) / np.sqrt(H1 * H2)
    model['b2'] = 0

    model['W3'] = np.random.randn(H2, C) / np.sqrt(H2 * C) 
    model['b3'] = 0

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
    dim, probobility = map_prameters
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

    # creat environment object
    env = sim_env(dim, probobility) 
    # initial state
    state = env.reset()

    if plotmap:
        plot_map(state, env.car_location)
        plt.show()
        plotmap = False

    step = 0
    collision = 0
    normal_reset = 0
    arrive = 0

    simu_round = 0

    # from 2d matrix to 1d vector
    state_vector = state.ravel()
    state_vector = state_vector.reshape(1, -1)
    # book keeping for one round and initialize
    state_round = []
    feedback_round = [] 
    action_round = []

    total_reward = 0
    reward_tr = []
    
    for i in range(max_iter):
        
        # predict action from state
        action = policy_forward(state_vector, model)
        state_round.append(state_vector) # keep state
        action_round.append(action) # keep action

        # do action
        state, feedback, terminate, status = env.step(action)
        state_vector = state.ravel()
        state_vector = state_vector.reshape(1, -1)

        feedback_round.append(feedback) # keep feedback
        total_reward += feedback

        if terminate:  # one simu_round finished
            # print status
            simu_round += 1
            
            # benchmark
            if status == 'collision':
                collision += 1.0
            elif status == 'arrive':
                arrive += 1.0
            else:
                normal_reset += 1.0
            if simu_round%100 ==0:
                # learning_rate *= decay
                # print "lr: ", learning_rate
                sum_all = collision + arrive + normal_reset
                print "collision: %i %%" %(collision/sum_all*100)
                print "arrive: %i %%" %(arrive/sum_all*100)
                print "normal_reset: %i %%" %(normal_reset/sum_all*100)
                print "reward/epoch: %2f" %total_reward
                reward_tr.append(total_reward)
                np.savetxt('reward_tr.txt', reward_tr)
                total_reward = 0
                collision = 0
                arrive = 0
                normal_reset = 0

            # list to batch
            state_batch = np.vstack(state_round)
            action_batch = np.vstack(action_round)
            feedback_batch = np.vstack(feedback_round)

            # redo mini batch forward to get cache
            cache, daction = policy_forward(state_batch, model, action_batch) 
            # minibatch get gradient
            gradient = policy_backward(feedback_batch, cache, daction, reg, model)

            # update weights
            rmsprop_cache,model = rmsprop_update(model, gradient, learning_rate, rmsprop_cache, decay)
    
            # reset to initial env
            step = 0
            state = env.reset()

            # reset batch
            state_round = []
            feedback_round = [] 
            daction_round = []


