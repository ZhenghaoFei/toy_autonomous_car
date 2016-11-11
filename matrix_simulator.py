<<<<<<< Updated upstream
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def map_create(map_matrix):
    return map_matrix

def plot_map(map_matrix, car_location):
    map_matrix[car_location] = 3 # use three to present car
    plt.imshow(map_matrix, interpolation='none')

def simulator(map_matrix, initial_car_location, goal_location, last_goaldistance, car_location = None, action = None, verbos=False):
    feedback = np.zeros(4) # default feedback 
    env =  np.zeros([10, 10])	
    env_distance = 3 # env use car as center, sensing distance
    map_env = np.pad(map_matrix, env_distance,'constant', constant_values=1)

    if action == None:
        car_location = initial_car_location
        car_x, car_y = car_location
        env_x = car_x + env_distance
        env_y = car_y + env_distance
        map_env[env_x, env_y] = 3
        env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
        return car_location, feedback, env

    # check if initial_location legal
    if map_matrix[car_location] == 1:
        print "initial position error"
        car_location = initial_car_location 
        return car_location, feedback, env

    # do action, move the car
    car_x, car_y = car_location
    if action == 0:
        car_x -= 1
        if verbos:    
            print("up")
    elif action == 1:
        car_x += 1
        if verbos:    
            print("down")
    elif action == 2:
        car_y += 1
        if verbos:    
            print("right")
    elif action == 3:
        car_y -= 1
        if verbos:    
            print("left")	
    car_location = car_x, car_y
    env_x = car_x + env_distance
    env_y = car_y + env_distance
    map_env[env_x, env_y] = 3
    env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
    goal_distance = np.sqrt(np.sum((np.asarray(goal_location) - np.asarray(car_location))**2)) # the distance from goal
    print goal_distance
    # check status
    if map_matrix[car_location] == 1:
        print "collision"
        feedback[action] = -1 # collision feedback
        car_location = initial_car_location # reset
        car_x, car_y = initial_car_location
        env_x = car_x + env_distance
        env_y = car_y + env_distance
        map_env[env_x, env_y] = 3
        env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

    elif map_matrix[car_location] == 0:
        # improve = last_goaldistance - goal_distance # whether approach goal
        # if improve > 0:
        #     feedback[action] = 0.01 # good moving feedback
        # elif improve < 0:
        #     feedback[action] = -0.5 # bad moving feedback
        feedback[action] = -0.1

    elif map_matrix[car_location] == 2:
        print "congratulations! You arrive destination"
        feedback[action] = 1 # get goal feedback

    return car_location, feedback, env, goal_distance

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
=======
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def map_create(map_matrix):
    return map_matrix

def plot_map(map_matrix, car_location):
    map_matrix[car_location] = 3 # use three to present car
    plt.imshow(map_matrix, interpolation='none')

def simulator(map_matrix, initial_car_location, goal_location, last_goaldistance, car_location = None, action = None):
    feedback = np.zeros(4) # default feedback 
    env =  np.zeros([10, 10])	
    env_distance = 3 # env use car as center, sensing distance
    map_env = np.pad(map_matrix, env_distance,'constant', constant_values=1)

    if action == None:
        car_location = initial_car_location
        car_x, car_y = car_location
        env_x = car_x + env_distance
        env_y = car_y + env_distance
        map_env[env_x, env_y] = 3
        env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
        return car_location, feedback, env

    # check if initial_location legal
    if map_matrix[car_location] == 1:
        print "initial position error"
        car_location = initial_car_location 
        return car_location, feedback, env

    # do action, move the car
    car_x, car_y = car_location
    if action == 0:
        car_x -= 1    
        print("up")
    elif action == 1:
        car_x += 1
        print("down")
    elif action == 2:
        car_y += 1
        print("right")
    elif action == 3:
        car_y -= 1
        print("left")	
    car_location = car_x, car_y
    env_x = car_x + env_distance
    env_y = car_y + env_distance
    map_env[env_x, env_y] = 3
    env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
    goal_distance = np.sqrt(np.sum((np.asarray(goal_location) - np.asarray(car_location))**2)) # the distance from goal
    print goal_distance
    # check status
    if map_matrix[car_location] == 1:
        print "collision"
        feedback[action] = -1 # collision feedback
        car_location = initial_car_location # reset
        car_x, car_y = initial_car_location
        env_x = car_x + env_distance
        env_y = car_y + env_distance
        map_env[env_x, env_y] = 3
        env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

    elif map_matrix[car_location] == 0:
        improve = last_goaldistance - goal_distance # whether approach goal
        feedback[action] = -0.05
        # if improve > 0:
        #     feedback[action] = 0.01 # good moving feedback
        # elif improve < 0:
        #     feedback[action] = -0.5 # bad moving feedback


    elif map_matrix[car_location] == 2:
        print "congratulations! You arrive destination"
        feedback[action] = 1 # get goal feedback

    return car_location, feedback, env, goal_distance

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
>>>>>>> Stashed changes
# # plt.show(fig)