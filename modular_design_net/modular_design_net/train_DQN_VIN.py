
import tensorflow as tf
import numpy as np
import tflearn
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
SUMMARY_DIR = './results_dqn_vin/'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
EVAL_EPISODES = 100


# ===========================
#   Q DNN
# ===========================
class QNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

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


    def create_Q_network(self):
        X = tflearn.input_data(shape=self.s_dim)
        k    = 10    # Number of value iterations performed
        ch_h = 150 # Channels in initial hidden layer 150
        ch_q = self.a_dim  # Channels in q layer (~actions) 10
        bias  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)
        # weights from inputs to q layer (~reward in Bellman equation)
        w0    = tf.Variable(np.random.randn(3, 3, 1, ch_h) * 0.01, dtype=tf.float32)
        w1    = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32)
        w     = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
        # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
        w_fb  = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)

        # output layers
        w_xq = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
        w_o   = tf.Variable(np.random.randn(ch_q,  self.a_dim ) * 0.01, dtype=tf.float32)

        # initial conv layer over image+reward prior

        h = tf.nn.conv2d(X, w0, name="h0", strides=(1, 1, 1, 1), padding='SAME') + bias

        r = tf.nn.conv2d(h, w1, name="r", strides=(1, 1, 1, 1), padding='SAME')
        q = tf.nn.conv2d(r, w, name="q" , strides=(1, 1, 1, 1), padding='SAME')
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

        for i in range(0, k-1):
            rv = tf.concat([r, v], 3)
            wwfb = tf.concat([w, w_fb], 2)

            q = tf.nn.conv2d(rv, wwfb, name="q", strides=(1, 1, 1, 1), padding='SAME')
            v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
        # do one last convolution
        q = tf.nn.conv2d(tf.concat([r, v], 3),
                            tf.concat([w, w_fb], 2), name="q", strides=(1, 1, 1, 1), padding='SAME')
        xq = tf.concat([X, q], 3)
        xq_flat = tf.contrib.layers.flatten(xq)
        q = tf.contrib.layers.fully_connected(xq_flat, 128, activation_fn=tf.nn.relu)
        out = tf.contrib.layers.fully_connected(q, self.a_dim, activation_fn=None)

        # out = tflearn.fully_connected(net, self.a_dim)
        return X, out

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
            Q_LEARNING_RATE, TAU)


        train(sess, env, Qnet, global_step)

if __name__ == '__main__':
    tf.app.run()
