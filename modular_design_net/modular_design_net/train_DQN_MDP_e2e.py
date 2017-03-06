# In this version of DQN, we explicity add MDP module 
# and do internal bellman equation with k step based on it
# test results: 8*8 60-70% after 500k

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
# import matplotlib.pyplot as plt
import time

from replay_buffer import ReplayBuffer
from simulator_gymstyle_old import *

# ==========================
#   Training Parameters
# ==========================

# Max episode length    
MAX_EP_STEPS = 100

# Base learning rate for the Qnet Network
Q_LEARNING_RATE = 1e-3
# Discount factor 
GAMMA = 0.9
# Soft target update param
TAU = 0.001
TARGET_UPDATE_STEP = 100

MINIBATCH_SIZE = 256
SAVE_STEP = 10000
EPS_MIN = 0.05
EPS_DECAY_RATE = 0.9999
# ===========================
#   Utility Parameters
# ===========================
# map size
MAP_SIZE  = 8
PROBABILITY = 0.1
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results_dqn_newimp/'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
EVAL_EPISODES = 100

# ===========================
#   MDP DNN
# ===========================
def conv2d_relu(X, W, bias=0):
    net = tf.nn.conv2d(X, W, strides=(1, 1, 1, 1), padding='SAME') + bias
    return tf.nn.relu(net)
# ===========================
#   Q DNN
# ===========================
class QNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        # Create the Qnet network
        self.inputs, self.out = self.create_Q_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_Q_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params)):]


        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. -self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.observed_q_value = tf.placeholder(tf.float32, [None])
        self.action_taken = tf.placeholder(tf.float32, [None, self.a_dim])
        self.predicted_q_value = tf.reduce_sum(tf.multiply(self.out, self.action_taken), reduction_indices = 1) 

        # Define loss and optimization Op
        self.Qnet_global_step = tf.Variable(0, name='Qnet_global_step', trainable=False)

        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.observed_q_value))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.Qnet_global_step)

    # def create_Q_MDP_network():
    #     state = tf.placeholder(tf.float32, shape = ([None] + list(self.s_dim)))
    #     r, next_s = create_MDP_network()

    # def MDP_network(self, w_MDP):


    def Q_network(self, input):
        net = conv2d_relu(input, self.w_Q1, bias=self.bias_Q)
        net = layers.max_pool2d(net, kernel_size=[2, 2], padding = 'SAME')
        net = conv2d_relu(net, self.w_Q2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, num_outputs=256, activation_fn=tf.nn.relu)
        out = layers.fully_connected(net, num_outputs=self.a_dim, activation_fn=None)
        return out


    def create_Q_network(self):
        state = tf.placeholder(tf.float32, shape = ([None] + list(self.s_dim)))
        gamma = tf.constant(self.gamma, tf.float32)
        # weights for MDP module
        m_ch = 150
        self.bias_MDP  = tf.Variable(np.random.randn(1, 1, 1, m_ch)    * 0.01, dtype=tf.float32)
        self.w_MDP_h = tf.Variable(np.random.randn(3, 3, 1, m_ch) * 0.01, dtype=tf.float32)
        self.w_MDP_snext_a = tf.Variable(np.random.randn(1, 1, m_ch, self.a_dim)    * 0.01, dtype=tf.float32)
        # weights for Q network
        q_ch1 = 32
        q_ch2 = 64
        self.bias_Q  = tf.Variable(np.random.randn(1, 1, 1, q_ch1)    * 0.01, dtype=tf.float32)
        self.w_Q1  = tf.Variable(np.random.randn(3, 3, 1, q_ch1)    * 0.01, dtype=tf.float32)
        self.w_Q2  = tf.Variable(np.random.randn(3, 3, q_ch1, q_ch2)    * 0.01, dtype=tf.float32)

        # Q
        Q0 = self.Q_network(state)
        a0 = tf.argmax(Q0, axis=1)
        a0 = tf.cast(a0, dtype=tf.int32)

        # # Combine action and state_action to get next state
        # t_a = tflearn.fully_connected(a0, 64)
        # t_sa = tflearn.fully_connected(MDP_snext_a, 64)
        # next_s = tflearn.activation(tf.matmul(net,t_a.W) + tf.matmul(action, t_sa.W) + net.b + t_sa.b, activation='relu')
        # print('a0', a0.shape)

        # MDP
        # next s
        MDP_h = conv2d_relu(state, self.w_MDP_h, bias=self.bias_MDP)
        # next s for all possible actions
        MDP_snext_a = conv2d_relu(MDP_h, self.w_MDP_snext_a)

        # MDP_snext_a = tf.reshape(MDP_snext_a, [MINIBATCH_SIZE * self.a_dim]+ list(self.s_dim), name=None)

        print('MDP_snext_a:', MDP_snext_a.shape)

        MDP_snext_a = tf.transpose(MDP_snext_a, perm=[0, 3, 1, 2]) # put channel at second
        a_idx = tf.stack([tf.range(0, tf.shape(a0)[0]), a0], axis=1)
        print('a_idx:', a_idx.shape)

        S1 = tf.gather_nd(MDP_snext_a, a_idx)
        S1 = tf.reshape(S1, shape=[tf.shape(a0)[0], self.s_dim[0], self.s_dim[1], self.s_dim[2]] )
        print('S1:', S1.shape)

        # reward for current state
        MDP_r = layers.flatten(MDP_h)
        MDP_r = layers.fully_connected(MDP_r, num_outputs=128, activation_fn=tf.nn.relu)
        r0 = layers.fully_connected(MDP_r, num_outputs=1, activation_fn=None)

        Q1 = self.Q_network(S1)


        # bellman equation
        Q_out = tf.add(r0, tf.multiply(Q1, gamma))

        return state, Q_out

    def train(self, inputs, action, observed_q_value):

        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action_taken: action,
            self.observed_q_value: observed_q_value
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    success_rate = tf.Variable(0.)
    tf.summary.scalar('Success Rate', success_rate)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar('Qmax Value', episode_ave_max_q)

    summary_vars = [success_rate, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, Qnet, global_step):

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

    # Initialize target network weights
    Qnet.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    i = global_step.eval()


    eval_acc_reward = 0
    tic = time.time()
    eps = 1
    while True:
        i += 1
        eps *= EPS_DECAY_RATE
        eps = max(eps, EPS_MIN)
        s = env.reset()
        # plt.imshow(s, interpolation='none')
        # plt.show()
        # s = prepro(s)
        ep_ave_max_q = 0

        if i % SAVE_STEP == 0 : # save check point every 1000 episode
            sess.run(global_step.assign(i))
            save_path = saver.save(sess, SUMMARY_DIR + "model.ckpt" , global_step = global_step)
            print("Model saved in file: %s" % save_path)
            print("Successfully saved global step: ", global_step.eval())


        for j in xrange(MAX_EP_STEPS):
            predicted_q_value = Qnet.predict(np.reshape(s, np.hstack((1, Qnet.s_dim))))
            predicted_q_value = predicted_q_value[0]

            np.random.seed()

            action = np.argmax(predicted_q_value)
            if np.random.rand() < eps:
                action = np.random.randint(4)
                # print('eps')
            # print'actionprob:', action_prob

            # print(action)
            # print(a)

            s2, r, terminal, info = env.step(action)
            # print r, info
            # plt.imshow(s2, interpolation='none')
            # plt.show()

            # s2 = prepro(s2)

            # print(np.reshape(s, (actor.s_dim,)).shape)
            action_vector = action_ecoder(action, Qnet.a_dim)
            replay_buffer.add(np.reshape(s, (Qnet.s_dim)), np.reshape(action_vector, (Qnet.a_dim)), r, \
                terminal, np.reshape(s2, (Qnet.s_dim)))

            s = s2
            eval_acc_reward += r

            if terminal:
                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > MINIBATCH_SIZE:     
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # Calculate targets
                    target_q = Qnet.predict_target(s2_batch)
                    y_i = []
                    for k in xrange(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * np.max(target_q[k]))

                    # # Update the Qnet given the target
                    predicted_q_value, _ = Qnet.train(s_batch, a_batch, y_i)
                
                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient

                # Update target networks every 1000 iter
                # if i%TARGET_UPDATE_STEP == 0:
                    Qnet.update_target_network()

                if i%EVAL_EPISODES == 0:
                    # summary
                    time_gap = time.time() - tic
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: (eval_acc_reward+EVAL_EPISODES)/2,
                        summary_vars[1]: ep_ave_max_q / float(j+1),
                    })
                    writer.add_summary(summary_str, i)
                    writer.flush()

                    print ('| Success: %i %%' % ((eval_acc_reward+EVAL_EPISODES)/2), "| Episode", i, \
                        '| Qmax: %.4f' % (ep_ave_max_q / float(j+1)), ' | Time: %.2f' %(time_gap), ' | Eps: %.2f' %(eps))
                    tic = time.time()

                    # print(' 100 round reward: ', eval_acc_reward)
                    eval_acc_reward = 0

                break


def prepro(state):
    """ prepro state to 3D tensor   """
    # print('before: ', state.shape)
    state = state.reshape(state.shape[0], state.shape[1], 1)
    # print('after: ', state.shape)
    # plt.imshow(state, interpolation='none')
    # plt.show()
    # state = state.astype(np.float).ravel()
    return state

def action_ecoder(action, action_dim):
    action_vector = np.zeros(action_dim, dtype=np.float32)
    action_vector[action] = 1
    return action_vector


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        global_step = tf.Variable(0, name='global_step', trainable=False)

        env = sim_env(MAP_SIZE, PROBABILITY) # creat 
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)

        # state_dim = np.prod(env.observation_space.shape)
        state_dim = [env.state_dim[0], env.state_dim[1], 1]
        print('state_dim:',state_dim)
        action_dim = env.action_dim
        print('action_dim:',action_dim)


        Qnet = QNetwork(sess, state_dim, action_dim, \
            Q_LEARNING_RATE, TAU, GAMMA)


        train(sess, env, Qnet, global_step)

if __name__ == '__main__':
    tf.app.run()
