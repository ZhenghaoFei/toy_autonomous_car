import numpy as np
from matrix_simulator import *
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from simulator_gymstyle import *

gamma = 0.9# discount factor for reward

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=float)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t] 
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(state, model, epsilon = None, action=None):
    Conv1_W1, Conv1_b1 = model['Conv1_W1'], model['Conv1_b1']
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3'] 

    filter_size = Conv1_W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    conv1_out, conv1_cache = conv_relu_forward(state, Conv1_W1, Conv1_b1, conv_param)
    layer1_out, layer1_cache = affine_relu_forward(conv1_out, W1, b1)
    layer2_out, layer2_cache = affine_relu_forward(layer1_out, W2, b2)
    action_scores, layer3_cache = affine_forward(layer2_out, W3, b3)


    if action is None:
        # action predction mode
        probs = np.exp(action_scores - np.max(action_scores))
        probs /= np.sum(probs)
        np.random.seed()
        dice = np.random.uniform() # roll the dice!
        probs = probs[0]
        # print probs
        if dice > epsilon:
            action = np.argmax(probs)
        else:
            action = np.random.choice(4, 1, p = probs)
            action = action[0]
        return action

    else:
        # print action
        # train mode
        loss, daction = softmax_loss(action_scores, action)
        daction = -daction
        cache = conv1_cache, layer1_cache, layer2_cache, layer3_cache

        # probs = np.exp(action_scores - np.max(action_scores, axis=1, keepdims=True))
        # probs /= np.sum(probs, axis=1, keepdims=True)
        # N = action_scores.shape[0]
        # daction = probs.copy()
        # daction[np.arange(N), action] -= 1 # fake label
        # # print('daction : ', daction)
        # # print action
        # daction /= N
        # daction = -daction # grad that encourages the action that was taken to be taken if feedback > 0 
        # print('daction : ', daction)
        return cache, daction

        # print "dx"
    # # print dx

# reward should be a vector e.g [0, 0, 1, 0]
def policy_backward(feedback, cache, daction, reg, model):

    # pre process feedback and daction
    dr = discount_rewards(feedback) # discouted rewards for this round, shape: step
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    # if len(dr) > 1:
    #     dr -= np.mean(dr)
    #     dr /= np.std(dr)

    # print('daction shape: ', daction.shape)
    # print('dr shape: ', dr.shape)
    # print('daction : ', daction)
    # print('dr : ', dr)
    daction *= dr # magic of PG

    # print('daction2 : ', daction)

    # """ backward pass.  """
    # unpack cache
    conv1_cache, layer1_cache, layer2_cache, layer3_cache = cache

    dlayer3 = affine_backward(daction, layer3_cache)
    dlayer2 = affine_relu_backward(dlayer3[0], layer2_cache)
    dlayer1 = affine_relu_backward(dlayer2[0], layer1_cache)
    dcon1 = conv_relu_backward(dlayer1[0], conv1_cache)

    grads = {}
    grads['Conv1_W1'] = dcon1[1] + 0.5 * reg * 2 * model['Conv1_W1']
    grads['Conv1_b1'] = dcon1[2]
    grads['W1'] = dlayer1[1] + 0.5 * reg * 2 * model['W1']
    grads['b1'] = dlayer1[2] 
    grads['W2'] = dlayer2[1] + 0.5 * reg * 2 * model['W2']
    grads['b2'] = dlayer2[2]
    grads['W3'] = dlayer3[1] + 0.5 * reg * 2 * model['W3']
    grads['b3'] = dlayer3[2]

    return grads


def sgd_update(model, gradient, learning_rate):
    for k,v in model.iteritems():
        model[k] += learning_rate * gradient[k]

def rmsprop_update(model, gradient, learning_rate, rmsprop_cache, decay_rate = 0.99):
    for k,v in model.iteritems():
        g = gradient[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
    return rmsprop_cache,model