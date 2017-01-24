
import tensorflow as tf
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
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 1e-4
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 1e-4
# Discount factor 
GAMMA = 0.9
# Soft target update param
TAU = 0.01
TARGET_UPDATE_STEP = 100
SAVE_STEP = 10000
EPS = 0.0
# ===========================
#   Utility Parameters
# ===========================
# map size
MAP_SIZE  = 2
PROB = 0
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
                tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.actor_global_step = tf.Variable(0, name='actor_global_step', trainable=False)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params), global_step=self.actor_global_step)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self): 
        inputs = tflearn.input_data(shape=self.s_dim)
        # net = tflearn.conv_2d(inputs, 8, 3, activation='relu', name='conv1')
        # net = tflearn.conv_2d(net, 16, 3, activation='relu', name='conv2')
        net = tflearn.fully_connected(inputs, 64, activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net)


        # net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(net, self.a_dim, activation='softmax', weights_init=w_init)
        out = tflearn.fully_connected(net, self.a_dim, activation='softmax')

        return inputs, out, out 

    def train(self, inputs, a_gradient):
        # print("actor global step: ", self.actor_global_step.eval())

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. -self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.critic_global_step = tf.Variable(0, name='critic_global_step', trainable=False)

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.critic_global_step)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=self.s_dim)
        action = tflearn.input_data(shape=[None,self.a_dim])
        # net = tflearn.conv_2d(inputs, 8, 3, activation='relu', name='conv1')
        # net = tflearn.conv_2d(net, 16, 3, activation='relu', name='conv2')
        net = tflearn.fully_connected(inputs, 64, activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net)

        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    episode_reward = tf.Variable(0.)
    tf.summary.scalar('Reward', episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar('Qmax Value', episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, global_step):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())

    # load model if have
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("./results")
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        print("global step: ", global_step.eval())

    else:
        print ("Could not find old network weights")

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    i = global_step.eval()

    ep_reward = 0
    ep_ave_max_q = 0
    tic = time.time()

    while True:
        i += 1
        s = env.reset()
        # plt.imshow(s, interpolation='none')
        # plt.show()
        s = prepro(s)
        j = 0


        if i % SAVE_STEP == 0 : # save check point every 1000 episode
            sess.run(global_step.assign(i))
            save_path = saver.save(sess, "./results/model.ckpt" , global_step = global_step)
            print("Model saved in file: %s" % save_path)
            print("Successfully saved global step: ", global_step.eval())


        while True:
            j += 1
            a = actor.predict(np.reshape(s, np.hstack((1, actor.s_dim))))
            a = a[0]
            action_prob = a

            np.random.seed()

            action = np.random.choice(actor.a_dim, 1, p = action_prob)
            action = action[0]
            if np.random.rand() < EPS:
                action = np.random.randint(4)
            # print'actionprob:', action_prob

            # print(action)
            # print(a)

            s2, r, terminal, info = env.step(action)
            # print r, info
            # plt.imshow(s2, interpolation='none')
            # plt.show()

            s2 = prepro(s2)

            # print(np.reshape(s, (actor.s_dim,)).shape)
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, \
                terminal, np.reshape(s2, (actor.s_dim,)))
            
            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:     
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
            
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)                
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

            # Update target networks every 1000 iter
            # if i%TARGET_UPDATE_STEP == 0:
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                # print('reward = ', r)
                break

        if i%100 == 0:
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: ep_reward,
                summary_vars[1]: ep_ave_max_q / float(j)
            })


            writer.add_summary(summary_str, i)
            writer.flush()
            time_gap = time.time() - tic
            # print(time_gap)
            print ('| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                '| Qmax: %.4f' % (ep_ave_max_q / float(j)), ' | Time: %.2f' %(time_gap))
            tic = time.time()

            ep_reward = 0
            ep_ave_max_q = 0


def prepro(state):
    """ prepro state to 3D tensor   """
    # print('before: ', state.shape)
    state = state.reshape(state.shape[0], state.shape[1], 1)
    # print('after: ', state.shape)
    # plt.imshow(state, interpolation='none')
    # plt.show()
    # state = state.astype(np.float).ravel()
    return state


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        global_step = tf.Variable(0, name='global_step', trainable=False)

        env = sim_env(MAP_SIZE, PROB) # creat 
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)

        # state_dim = np.prod(env.observation_space.shape)
        state_dim = [env.state_dim[0], env.state_dim[1], 1]
        print('state_dim:',state_dim)
        action_dim = env.action_dim
        print('action_dim:',action_dim)


        actor = ActorNetwork(sess, state_dim, action_dim, \
            ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, \
            CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())


        train(sess, env, actor, critic, global_step)

if __name__ == '__main__':
    tf.app.run()
