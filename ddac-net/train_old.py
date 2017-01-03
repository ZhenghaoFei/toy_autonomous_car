""" 
Implementation of DDPG - Pong - Deep Deterministic Policy Gradient

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
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.09
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.9
# Soft target update param
TAU = 0.001
EPSILON = 0.9 # eps greedy

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = "Pong-v0"
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100
MINIBATCH_SIZE = 1

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
        # self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params)

    def create_actor_network(self): 
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 50, activation='relu')
        net = tflearn.fully_connected(net, 100, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='relu', weights_init=w_init)
        scaled_out = tf.nn.softmax(out, name='action_prob')
        # scaled_out = tf.mul(out, self.action_bound) # Scale output to -action_bound to action_bound
        return inputs, out, scaled_out 

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })


    def get_num_trainable_vars(self):
        return self.num_trainable_vars

def discount_rewards(r):

    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=float)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.merge_all_summaries()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

    # # Initialize target network weights
    # actor.update_target_network()
    # critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    ep_reward = 0
    ep_ave_max_q = 0

    for i in xrange(MAX_EPISODES):

        s = env.reset()
        s = s.reshape(-1,)


        j = 0

        s2_batch = []
        r_batch = []

        while True:
            j += 1


            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            a = a[0]
            print'actionprob:', a
            dice = np.random.uniform() # roll the dice!
            if dice < EPSILON:
                action = np.argmax(a)
            else:
                action = np.random.choice(4, 1, p = a)

            s2, r, terminal, info = env.dostep(action)
            s2 = s2.reshape(-1,)
            # print(s2.shape)
            s = s2
            ep_reward += r

            s2_batch.append(s2)
            r_batch.append(r)


            if terminal:
                epr = np.vstack(r_batch)
                # print epr

                dr = discount_rewards(epr) # discouted rewards for this round, shape: step
                # print dr

                # if len(dr) > 1:
                #     # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                #     dr -= np.mean(dr)
                #     dr /= np.std(dr)
                feedback_input = np.zeros([len(dr),env.action_dim]) # shape: step, 4
                for k in range(len(dr)): # transfer reward vector to reward matrix
                    feedback_input[k,:] = dr[k]

                actor.train(s2_batch, feedback_input)



                break
        if i%100 == 0:
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: ep_reward,
                summary_vars[1]: ep_ave_max_q / float(j)
            })


            writer.add_summary(summary_str, i)
            writer.flush()

            print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                '| Qmax: %.4f' % (ep_ave_max_q / float(j))

            ep_reward = 0
            ep_ave_max_q = 0

def main(_):
    with tf.Session() as sess:
        
        env = sim_env(10, 0)
        # np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        # env.seed(RANDOM_SEED)

        # state_dim = np.prod(env.observation_space.shape)
        state_dim = env.state_dim
        print('state_dim:',state_dim)
        action_dim = env.action_dim
        print('action_dim:',action_dim)
        # # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        # actor = ActorNetwork(sess, state_dim, action_dim, action_bound, \
        #     ACTOR_LEARNING_RATE, TAU)

        actor = ActorNetwork(sess, state_dim, action_dim, ACTOR_LEARNING_RATE, TAU)

        train(sess, env, actor)

        # critic = CriticNetwork(sess, state_dim, action_dim, \
        #     CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        # if GYM_MONITOR_EN:
        #     if not RENDER_ENV:
        #         env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
        #     else:
        #         env.monitor.start(MONITOR_DIR, video_callable=False, force=True)

        # train(sess, env, actor, critic)

        # if GYM_MONITOR_EN:
        #     env.monitor.close()



if __name__ == '__main__':
    tf.app.run()
