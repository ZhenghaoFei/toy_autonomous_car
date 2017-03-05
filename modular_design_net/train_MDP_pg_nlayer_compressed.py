
import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import tflearn
import matplotlib.pyplot as plt
import time

from replay_buffer import ReplayBuffer
from simulator_gymstyle_old import *

# ==========================
#   Training Parameters
# ==========================

# Max episode length    
MAX_EP_STEPS = 100
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 1e-4
LAYERS = 5

# Discount factor 
GAMMA = 0.9
# Soft target update param
TAU = 0.001
TARGET_UPDATE_STEP = 100

MINIBATCH_SIZE = 512
SAVE_STEP = 50000
EPS_MIN = 0.05
EPS_DECAY_RATE = 0.9999
# ===========================
#   Utility Parameters
# ===========================
# map size
MAP_SIZE  = 5
PROBABILITY = 0.1
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results_pg_mdp_' + str(LAYERS) + "l_com/"
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 1000000
EVAL_EPISODES = 100

# ===========================
#   MDP DNN
# ===========================
def conv2d_relu(X, W, bias=0):
    net = tf.nn.conv2d(X, W, strides=(1, 1, 1, 1), padding='SAME') + bias
    return tf.nn.relu(net)

# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a policy.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, layers):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.gamma = gamma
        self.layers = layers
        self.learning_rate = learning_rate

        # Actor Network
        self.inputs, self.actions_out, self.log_prob = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        
        # Compute the loss here 
        self.actions_in = tf.placeholder(tf.int32)
        self.advantages = tf.placeholder(tf.float32)
        indices = tf.stack([tf.range(0, MINIBATCH_SIZE), self.actions_in], axis=1)
        act_prob = tf.gather_nd(self.log_prob, indices)
        loss = -tf.reduce_sum(tf.multiply(act_prob, self.advantages))

        # Optimization Op
        self.actor_global_step = tf.Variable(0, name='actor_global_step', trainable=False)

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(loss)

    def MDP_network(self, input, action, feature_dim):
        # hidden layer
        MDP_h = tf.nn.relu(tf.add(tf.matmul(input, self.w_MDP_h), self.bias_MDP))

        # state transition
        MDP_snext_a = tf.nn.relu(tf.matmul(MDP_h, self.w_MDP_snext_a))
        MDP_snext_a = tf.reshape(MDP_snext_a, [tf.shape(action)[0], feature_dim, self.a_dim])
        MDP_snext_a = tf.transpose(MDP_snext_a, perm=[0, 2, 1]) # put channel at second
        a_idx = tf.stack([tf.range(0, tf.shape(action)[0]), action], axis=1)
        S_next = tf.gather_nd(MDP_snext_a, a_idx)

        # reward for current state
        MDP_r1 = tf.nn.relu(tf.matmul(MDP_h, self.w_MDP_r1))
        MDP_r2 = tf.nn.relu(tf.matmul(MDP_r1, self.w_MDP_r2))
        r_current = tf.nn.relu(tf.matmul(MDP_r2, self.w_MDP_r_out))

        return S_next, r_current

    def Q_network(self, state):
        net = tf.nn.relu(tf.add(tf.matmul(state, self.w_Q1), self.bias_Q))
        net = tf.nn.relu(tf.matmul(net, self.w_Q2))
        out = tf.nn.relu(tf.matmul(net, self.w_Qout))
        return out

    def sensor_network(self, input, feature_dim):
        # compress input state into feature dim
        net = tflearn.conv_2d(input, 128, 3, activation='relu', name='conv1')
        net = tflearn.layers.conv.max_pool_2d (net, 2, strides=None, padding='same', name='MaxPool2D1')
        net = tflearn.conv_2d(net, 64, 2, activation='relu', name='conv2')
        net = tflearn.fully_connected(net, 128, activation='relu')
        out = tflearn.fully_connected(net, feature_dim, activation='relu')
        return out

    def create_actor_network(self): 
        state = tf.placeholder(tf.float32, shape = ([None] + list(self.s_dim)))
        gamma = tf.constant(self.gamma, tf.float32)
        Q_out = tf.zeros([tf.shape(state)[0], self.a_dim], tf.float32)
        feature_dim = 128
        # weights for MDP module
        m_ch = 150
        m_ch_r1 = 128
        m_ch_r2 = 64
        self.bias_MDP  = tf.Variable(np.random.randn(1, m_ch)    * 0.01, dtype=tf.float32)
        self.w_MDP_h = tf.Variable(np.random.randn(feature_dim, m_ch) * 0.01, dtype=tf.float32)
        self.w_MDP_snext_a = tf.Variable(np.random.randn(m_ch, feature_dim * self.a_dim)    * 0.01, dtype=tf.float32)
        self.w_MDP_r1 = tf.Variable(np.random.randn(m_ch, m_ch_r1)    * 0.01, dtype=tf.float32)
        self.w_MDP_r2 = tf.Variable(np.random.randn(m_ch_r1, m_ch_r2)    * 0.01, dtype=tf.float32)
        self.w_MDP_r_out = tf.Variable(np.random.randn(m_ch_r2, 1)    * 0.01, dtype=tf.float32)

        # weights for Q network
        q_ch1 = 32
        q_ch2 = 64
        self.bias_Q  = tf.Variable(np.random.randn(1, q_ch1)    * 0.01, dtype=tf.float32)
        self.w_Q1  = tf.Variable(np.random.randn(feature_dim, q_ch1)    * 0.01, dtype=tf.float32)
        self.w_Q2  = tf.Variable(np.random.randn(q_ch1, q_ch2)    * 0.01, dtype=tf.float32)
        self.w_Qout  = tf.Variable(np.random.randn(q_ch2, self.a_dim)    * 0.01, dtype=tf.float32)

        # features
        state_feature = self.sensor_network(state, feature_dim)
        for i in range(self.layers):
            # Q
            Q = self.Q_network(state_feature)
            a = tf.argmax(Q, axis=1)
            a = tf.cast(a, dtype=tf.int32)
            # MDP
            state_feature, r0 = self.MDP_network(state_feature, a, feature_dim)
            Q_next = self.Q_network(state_feature)
            # bellman equation
            Q_out = tf.add(Q_out, tf.multiply(r0, tf.pow(gamma, i)))

        Q_out = tf.add(r0, tf.multiply(Q_next, tf.pow(gamma, self.layers-1)))

        # polocy net
        net = tflearn.fully_connected(Q_out, 32, activation='relu')
        net = tflearn.fully_connected(net, self.a_dim, activation='relu')

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        logits = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)
        actions_out = tf.reshape(tf.multinomial(logits, 1), [])
        log_prob = tf.log(tf.nn.softmax(logits))

        return state, actions_out, log_prob

    def train(self, states, actions_in, advantages):
        # print("actor global step: ", self.actor_global_step.eval())

        self.sess.run(self.optimize, feed_dict={
            self.inputs: states,
            self.actions_in: actions_in,
            self.advantages: advantages
        })

    def predict(self, inputs):
        return self.sess.run(self.actions_out, feed_dict={
            self.inputs: inputs
        })



# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    success_rate = tf.Variable(0.)
    tf.summary.scalar('Success_Rate', success_rate)

    summary_vars = [success_rate]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, global_step):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())

    # load model if have
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SUMMARY_DIR)
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        print("global step: ", global_step.eval())

    else:
        print ("Could not find old network weights")

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    i = global_step.eval()


    eval_acc_reward = 0
    tic = time.time()
    eps = 1

    while True:
        i += 1
        s = env.reset()
        ep_ave_max_q = 0
        eps *= EPS_DECAY_RATE
        eps = max(eps, EPS_MIN)

        episode_s, episode_acts, episode_rewards = [], [], []

        if i % SAVE_STEP == 0 : # save check point every 1000 episode
            sess.run(global_step.assign(i))
            save_path = saver.save(sess, SUMMARY_DIR + "model.ckpt" , global_step = global_step)
            print("Model saved in file: %s" % save_path)
            print("Successfully saved global step: ", global_step.eval())


        for j in xrange(MAX_EP_STEPS):

            # print(s.shape)

            # Added exploration noise

            action = actor.predict(np.reshape(s, np.hstack((1, actor.s_dim))))
            # print action

            s2, r, terminal, info = env.step(action)
            # plt.imshow(s2, interpolation='none')
            # plt.show()
            episode_s.append(s)
            episode_acts.append(action)
            episode_rewards.append(r)

            s = s2
            eval_acc_reward += r

            if terminal:
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                episode_rewards = np.asarray(episode_rewards)
                # print('episode_rewards', episode_rewards)

                episode_rewards = discount_rewards(episode_rewards)
                # print('after', episode_rewards)
                # update buffer
                for n in range(len(episode_rewards)):
                    replay_buffer.add(np.reshape(episode_s[n], (actor.s_dim)), episode_acts[n],
                     episode_rewards[n], terminal, np.reshape(episode_s[n], (actor.s_dim)))
                
                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > MINIBATCH_SIZE:     
                    s_batch, a_batch, r_batch, t_batch, _ = replay_buffer.sample_batch(MINIBATCH_SIZE)
                    # Update the actor policy using the sampled gradient
                    actor.train(s_batch, a_batch, r_batch)



                # print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                #     '| Qmax: %.4f' % (ep_ave_max_q / float(j+1))

                if i%EVAL_EPISODES == 0:
                    # summary
                    time_gap = time.time() - tic
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: (eval_acc_reward+EVAL_EPISODES)/2,
                    })
                    writer.add_summary(summary_str, i)
                    writer.flush()

                    print ('| Success: %i %%' % ((eval_acc_reward+EVAL_EPISODES)/2), "| Episode", i, \
                         ' | Time: %.2f' %(time_gap), ' | Eps: %.2f' %(eps))
                    tic = time.time()

                    # print(' 100 round reward: ', eval_acc_reward)
                    eval_acc_reward = 0

                break

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    discounted_r = discounted_r.astype(np.float32)
    running_add = 0

    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r)
    # discounted_r /= np.std(discounted_r + 1e-10)

    return discounted_r

def prepro(state):
    """ prepro state to 3D tensor   """
    # print('before: ', state.shape)
    state = state.reshape(state.shape[0], state.shape[1], 1)
    # print('after: ', state.shape)
    # plt.imshow(state, interpolation='none')
    # plt.show()
    # state = state.astype(np.float).ravel()
    return stat


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        global_step = tf.Variable(0, name='global_step', trainable=False)

        env = sim_env(MAP_SIZE, PROBABILITY) # creat  env

        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        # state_dim = np.prod(env.observation_space.shape)
        state_dim = [env.state_dim[0], env.state_dim[1], 1]
        print('state_dim:',state_dim)
        action_dim = env.action_dim
        print('action_dim:',action_dim)


        actor = ActorNetwork(sess, state_dim, action_dim, ACTOR_LEARNING_RATE, GAMMA, LAYERS)

        train(sess, env, actor, global_step)

if __name__ == '__main__':
    tf.app.run()
