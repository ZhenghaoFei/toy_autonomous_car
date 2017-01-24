import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mazegen import make_maze

FIX_STARTEND = True

class sim_env(object):
    def __init__(self, dim):
        self.dim = dim
        self.WALL_VALUE = 10
        self.CAR_VALUE = 100
        self.GOAL_VALUE = 200
        self.max_step = 100
        self.state_dim = [(self.dim + 2) , (self.dim + 2)]
        self.action_dim = 4

    def reset(self):
      # np.random.seed(3)
      self.map_matrix, self.start, self.goal = make_maze(self.dim, self.dim, self.WALL_VALUE)
      self.map_matrix[self.start] = 0

      self.car_location = self.start
      self.map_matrix[self.goal] = self.GOAL_VALUE
      self.current_step = 0

      # pad the environment edge as wall to prevent the car move out side
      self.map_env = np.pad(self.map_matrix, 1,'constant', constant_values=self.WALL_VALUE)
      temp_env = np.copy(self.map_env) # this env is for return with changing car location
      car_x, car_y = self.car_location
      env_x = car_x + 1
      env_y = car_y + 1
      env_car_location = env_x, env_y
      temp_env[env_car_location] = self.CAR_VALUE

      return temp_env


    def plot_map():
        # make a copy to prevent change in the real map
        plot_map = np.copy(self.map_env)
        plot_map[self.car_location] = self.CAR_VALUE
        plt.imshow(plot_map, interpolation='none')
        plt.show()


    def step(self, action):
        self.current_step += 1
        self.done = False
        temp_env = np.copy(self.map_env) # this env is for return with changing car location

        feedback = 0 # default feedback 
      
        # do action, move the car, 0-3 to represent move in four directions
        car_x, car_y = self.car_location

        if action == 0:
            car_x -= 1
        elif action == 1:
            car_x += 1
        elif action == 2:
            car_y += 1
        elif action == 3:
            car_y -= 1
        else:
            print('action error!')
    	
        self.car_location = car_x, car_y

        # since the environment has pad
        env_x = car_x + 1
        env_y = car_y + 1
        env_car_location = env_x, env_y

        # print "step: ", current_step

        # check status
        status = 'normal'
        if temp_env[env_car_location] == self.WALL_VALUE:
            # print "collision"
            feedback = -1 # collision feedback
            self.done = True
            status = 'collision'

        elif temp_env[env_car_location] == 0:
            if self.current_step >= self.max_step:
                # print "exceed max step"
                feedback = -1
                status = 'exceed max step'
                self.done = True
                # print "self.done"

        elif temp_env[env_car_location] == self.GOAL_VALUE:
            # print "congratulations! You arrive destination"
            feedback = 1 # get goal feedback
            self.done = True
            status = 'arrive'

        temp_env[env_car_location] = self.CAR_VALUE

        return temp_env, feedback, self.done, status
