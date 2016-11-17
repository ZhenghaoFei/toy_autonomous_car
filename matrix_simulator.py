import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

WALL_VALUE = 10
CAR_VALUE = 100
def random_map(dim1, dim2, propobility):
  # np.random.seed(3)
  map_matrix = np.zeros([dim1, dim2])
  for i in range(dim1):
    for j in range(dim2):
      a = np.random.random(1)
      if a < propobility:
        map_matrix[i,j] = WALL_VALUE
  start = np.random.random_integers(0, 9, 2)
  start_x = start[0]
  start_y = start[1]
  start = start_x, start_y
  map_matrix[start] = 0

  goal = np.random.random_integers(0, 9, 2)
  goal_x = goal[0]
  goal_y = goal[1]
  goal = goal_x, goal_y
  map_matrix[goal] = 200
  return map_matrix, start, goal

def plot_map(map_matrix, car_location):
    map_matrix[car_location] = CAR_VALUE# use three to present car
    plt.imshow(map_matrix, interpolation='none')

def simulator(map_matrix, initial_car_location, goal_location, last_goaldistance, step, max_step = 30, car_location = None, action = None, verbos=False):
    reset = False
    feedback = 0 # default feedback 
    env =  np.zeros([10, 10])	
    env_distance = 10 # env use car as center, sensing distance
    map_env = np.pad(map_matrix, env_distance,'constant', constant_values=WALL_VALUE)
    # map_env = np.copy(map_matrix)
    if action == None:
        car_location = initial_car_location
        car_x, car_y = car_location
        env_x = car_x + env_distance
        env_y = car_y + env_distance
        # env use car as center, sensing distance
        map_env[env_x, env_y] = CAR_VALUE
        env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
        return car_location, feedback, env

    # check if initial_location legal
    if map_matrix[car_location] == WALL_VALUE:
        print "initial position error"
        car_location = initial_car_location
        reset = True
        return car_location, feedback, map_env, goal_distance, step, reset

    # do action, move the car
    car_x, car_y = car_location

    if action == 0:
        car_x -= 1
    elif action == 1:
        car_x += 1
    elif action == 2:
        car_y += 1
    elif action == 3:
        car_y -= 1
	
    car_location = car_x, car_y
    env_x = car_x + env_distance
    env_y = car_y + env_distance
    env_location = env_x, env_y
    # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
    goal_distance = np.sqrt(np.sum((np.asarray(goal_location) - np.asarray(car_location))**2)) # the distance from goal
    # print "goal_distance: ", goal_distance
    step += 1
    # print "step: ", step
    # check status
    status = 'normal'
    if map_env[env_location] == WALL_VALUE:
        # print "collision"
        feedback = -1 # collision feedback
        reset = True
        status = 'collision'
        # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

    elif map_env[env_location] == 0:
        # improve = last_goaldistance - goal_distance # whether approach goal
        # if improve > 0:
        #     feedback = 0.01 # good moving feedback
        # elif improve < 0:
        #     feedback = -0.01 # bad moving feedback
        # # feedback = -0.08
        if step >= max_step:
            feedback = -1
            reset = True
            # print "reset"


    elif map_env[env_location] == 200:
        # print "congratulations! You arrive destination"
        feedback = 100 # get goal feedback
        reset = True
        status = 'arrive'
    map_env[env_location] = CAR_VALUE
    # map_env = map_env.ravel
    env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

    return car_location, feedback, env, goal_distance, step, reset, status

# map_matrix = np.array\
#      ([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  1.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  2.],
#        [ 0.,  0.,  0.,  1.,  1,  1.,  1.,  1.,  1.,  1.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

# initial_car_location = 1, 1 # initial car location x and y
# car_location = initial_car_location
# car_location_save =[]
# while True:
#     action = 2
#     car_location, feedback, env = simulator(map_matrix, initial_car_location, car_location, action)
#     car_location_save.append(car_location)
#     if feedback < 0:
#         print "game end"
#         break

# fig, ax = plt.subplots()  

# def animate(i):
#   map_plot = np.copy(map_matrix)
#   map_plot[car_location_save[i]] = 3
#   ax.imshow(map_plot, interpolation='none')

# anim = animation.FuncAnimation(fig, animate, frames = len(car_location_save))
# plt.show()
# # fig = plot_map(map_matrix, car_location)