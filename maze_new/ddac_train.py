""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym 
import tflearn

from replay_buffer import ReplayBuffer
from simulator_gymstyle import *

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 15000000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001

EVAL_EPISODES = 100

# ===========================
#   Utility Parameters
# ===========================

# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
SAVE_STEP = 1000
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
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound #?
        self.learning_rate = learning_rate
        self.tau = tau #?

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
        self.actor_global_step = tf.Variable(0, name='actor_global_step', trainable=False) #?

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params), global_step=self.actor_global_step)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self): 
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 100, activation='relu')
        net = tflearn.fully_connected(net, 50, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.mul(out, self.action_bound) # Scale output to -action_bound to action_bound
        return inputs, out, scaled_out #? tanh ??

    def train(self, inputs, a_gradient):
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
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
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
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 100, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 50)
        t2 = tflearn.fully_connected(action, 50)

        net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
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
    tf.scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Qmax Value", episode_ave_max_q)
    eval_acc_reward = tf.Variable(0.)
    tf.scalar_summary("eval_acc_reward", eval_acc_reward)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    summary_vars = [episode_reward, episode_ave_max_q, eval_acc_reward]
    summary_ops = tf.merge_all_summaries()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, global_step):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("./results/tf_ddpg")
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        print("global step: ", global_step.eval())
        # for v in tf.all_variables():
        #     print(v.name)


    else:
        print ("Could not find old network weights")
    i = global_step.eval()

    writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph) #?

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    eval_acc_reward = 0
    eval_acc_reward_summary = -100
    # for i in xrange(MAX_EPISODES):
    while 1:
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        if i % SAVE_STEP == 0 : # save check point every 1000 episode
            sess.run(global_step.assign(i))
            save_path = saver.save(sess, "./results/model.ckpt" , global_step = global_step)
            print("Model saved in file: %s" % save_path)
            print("Successfully saved global step: ", global_step.eval())

        i += 1


        for j in xrange(MAX_EP_STEPS):


            # print(s.shape)

            # Added exploration noise

            a = actor.predict(np.reshape(s, (1, -1)))
            action_score = a[0]
            probs = np.exp(action_score - np.max(action_score))
            # probs = np.exp(action_score)
            probs /= np.sum(probs)
            # np.random.seed()
            # epsilon = 0.5 # eps greedy
            # dice = np.random.uniform() # roll the dice!
            # if dice < epsilon:
            # action = np.argmax(probs)
            # else:
            action = np.random.choice(4, 1, p = probs)
            # print action
            s2, r, terminal, info = env.step(action)

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

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r
            eval_acc_reward += r
            if terminal:
                # print j
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j+1),
                    summary_vars[2]: eval_acc_reward_summary

                })
                writer.add_summary(summary_str, i)
                writer.flush()
                print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j+1))

                if i%EVAL_EPISODES == 0:
                    print(' 100 round reward: ', eval_acc_reward)
                    eval_acc_reward_summary = eval_acc_reward
                    eval_acc_reward = 0

                break

def main(_):
    with tf.Session() as sess:
        
        global_step = tf.Variable(0, name='global_step', trainable=False)

        env = sim_env(5, 0.1)
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)


        # state_dim = np.prod(env.observation_space.shape)
        state_dim = env.state_dim
        print('state_dim:',state_dim)
        action_dim = env.action_dim
        print('action_dim:',action_dim)
        action_bound = 1

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, \
            ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, \
            CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())


        train(sess, env, actor, critic, global_step)

if __name__ == '__main__':
    tf.app.run()
