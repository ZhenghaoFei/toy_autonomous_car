# import tensorflow as tf
import numpy as np
# import gym 
# import tflearn
import matplotlib.pyplot as plt

print (np.random.rand(1))

# # Render gym env during training
# RENDER_ENV = True
# # Use Gym Monitor
# GYM_MONITOR_EN = False
# # Gym environment
# ENV_NAME = 'CarRacing-v0'
# # Directory for storing gym results
# MONITOR_DIR = './results/gym_ddpg'
# # Directory for storing tensorboard summary results
# SUMMARY_DIR = './results/tf_ddpg'
# RANDOM_SEED = 1234
# # Size of replay buffer
# BUFFER_SIZE = 10000
# MINIBATCH_SIZE = 20


# def prepro(s):
#   """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#   s = s[:84,:,:] # crop
#   s = s[::2,::2,1]   # downsample by factor of 2
#   s[s == 255] = 2 # road mark ==2

#   s[ s>110] = 255 # grass
#   s[ s==105] = 1 # road
#   s[ s==107] = 1 # road
#   s[ s==102] = 1 # road
#   s[ s==0] = 100
#   s[ s==76] = 100

#   # I[I == 144] = 0 # erase background (background type 1)
#   # I[I == 109] = 0 # erase background (background type 2)
#   # I[I != 0] = 1 # everything else (paddles, ball) just set to 1
#   return s
# env = gym.make(ENV_NAME)
# env.ZOOM = 0

# env.ZOOM_FOLLOW = False

# s = env.reset()
# print s.shape
# s = prepro(s)
# env.render()
# # env.monitor.start(MONITOR_DIR, force=True)
# if GYM_MONITOR_EN:
#     if not RENDER_ENV:
#         env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
#     else:
#         env.monitor.start(MONITOR_DIR, force=True)


# if GYM_MONITOR_EN:
#     env.monitor.close()


# while 1:
# 	env.render()
# 	plt.imshow(s, interpolation='none')
# 	plt.show()
# 	a = [0, 0.5, 0]
# 	s, r, terminal, info = env.step(a)
# 	s = prepro(s)

# 	print s