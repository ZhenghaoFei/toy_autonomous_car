""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H1 = 50 # number of hidden layer neurons
H2 = 100
batch_size = 1 # every how many episodes to do a param update?
lr_decay = 0.9995 # decay factor for RMSProp leaky sum of grad^2
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
reg=1
debug = False

def creat_model(D, H1, H2, C=1):
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

# model initialization
D = (80) * (80) # input dimensionality, because 1 pad
model = creat_model(D, H1, H2)
if resume:
    print "model resumed"
    model = load_model(model)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

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
    p = sigmoid(action_score)

    # np.random.seed()

    # dice = np.random.uniform() # roll the dice!
    # # print dice
    # # print "probs:"
    # # print probs

    # # action = 0
    # # for i in range(probs.shape[0]):
    # #     prob = np.sum(probs[:i+1])
    # #     if dice < prob:
    # #         action = i
    # #         # print action
    # #         break  # if dice fall in certain range chose the action
    # # action = np.random.choice(2, 1, p = probs)
    # # print "action ", action


    # N = action_score.shape[0]
    # dx = probs.copy()
    # dx[action] -= 1 # fake label
    # dx /= N
    # dx = -dx # grad that encourages the action that was taken to be taken if feedback > 0 
    # # print "dx"
    # # print dx
    # print(p)
    return p, h1, h2 # return action, and hidden state

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

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,h1s,h2s,dlogps,drs = [],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h1, h2 = policy_forward(x, model)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  h1s.append(h1) # hidden state
  h2s.append(h2) # hidden state

  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph1 = np.vstack(h1s)
    eph2 = np.vstack(h2s)

    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,h1s,h2s,dlogps,drs = [],[],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(epdlogp, eph1, eph2, epx, model, reg)
    # grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
     print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')