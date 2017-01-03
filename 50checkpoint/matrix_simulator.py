import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

WALL_VALUE = 10
CAR_VALUE = 100
def random_map(dim1, dim2, propobility):
    fix_start_end = True
    # np.random.seed(3)
    map_matrix = np.zeros([dim1, dim2])
    for i in range(dim1):
        for j in range(dim2):
          a = np.random.random(1)
          if a < propobility:
            map_matrix[i,j] = WALL_VALUE
    if fix_start_end:
        start = np.array([0, 0])
    else:
        start = np.random.random_integers(0, dim1-1, 2)
    start_x = start[0]
    start_y = start[1]
    start = start_x, start_y
    map_matrix[start] = 0
    if fix_start_end:
        goal = np.array([3, 3])
    else:
        goal = np.random.random_integers(0, dim1-1, 2)
    goal_x = goal[0]
    goal_y = goal[1]
    goal = goal_x, goal_y
    map_matrix[goal] = 200
    return map_matrix, start, goal

def maze_gen(mx,my):
    maze = np.zeros((my, mx))
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    # start the maze from (0, 0)
    stack = [(np.random.randint(0, mx - 1), np.random.randint(0, my - 1))]#[(0, 0)]
    start = stack[0]
   
    while len(stack) > 0:
        (cx, cy) = stack[-1]
        maze[cy][cx] = WALL_VALUE
        # find a new cell to add
        nlst = [] # list of available neighbors
        for i in range(4):
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                if maze[ny][nx] == 0:
            # of occupied neighbors must be 1
                    ctr = 0
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey][ex] == WALL_VALUE: ctr += 1
                    if ctr == 1: nlst.append(i)
        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = np.random.choice(nlst)
            cx += dx[ir]; cy += dy[ir]
            stack.append((cx, cy))
        else: stack.pop()

    maze = np.abs(maze - WALL_VALUE)
    # initalize the starting point
    #start = np.random.random_integers(0, mx-1, 2)
    #start = start[0], start[1]
    #while maze[start] == WALL_VALUE:
    #    start = (np.random.randint(mx), np.random.randint(my))
    #start_x = start[0]
    #start_y = start[1]
    #start = start_x, start_y
    # map_matrix[start] = 0

    goal = np.random.random_integers(0, mx-1, 2)
    goal = goal[0], goal[1]
    while maze[goal] == WALL_VALUE or (goal[0]==start[0] and goal[1]==start[1]):
        goal = (np.random.randint(mx), np.random.randint(my))
    goal_x = goal[0]
    goal_y = goal[1]
    goal = goal_x, goal_y
    maze[goal] = 200

    return maze, start, goal  # transpose and invert 0s and 1s

def plot_map(map_matrix, car_location):
    map_matrix[car_location] = CAR_VALUE# use three to present car
    plt.imshow(map_matrix, interpolation='none', cmap='Greys')

def simulator(map_matrix, initial_car_location, goal_location, last_goaldistance, step, max_step = 20, car_location = None, action = None, verbos=False):
    reset = False
    feedback = 0 # default feedback 
    env =  np.zeros([10, 10])	
    env_distance = 1 # env use car as center, sensing distance
    map_env = np.pad(map_matrix, env_distance,'constant', constant_values=WALL_VALUE)
    # map_env = np.copy(map_matrix)
    if action == None:
        car_location = initial_car_location
        car_x, car_y = car_location
        env_x = car_x + env_distance
        env_y = car_y + env_distance
        # env use car as center, sensing distance
        map_env[env_x, env_y] = CAR_VALUE
        # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
        return car_location, feedback, map_env

    # check if initial_location legal
    if map_matrix[car_location] == WALL_VALUE:
        print "initial position error"
        # print("check car loc", car_location)
        # print(map_matrix)
        car_location = initial_car_location
        reset = True
        return car_location, feedback, map_env

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
        #     feedback = 0.001 # good moving feedback
        # elif improve < 0:
        #     feedback = -0.002 # bad moving feedback
        if step >= max_step:
            feedback = -1
            reset = True

            # print "reset"


    elif map_env[env_location] == 200:
        # print "congratulations! You arrive destination"
        feedback = 10 # get goal feedback
        reset = True
        status = 'arrive'
    map_env[env_location] = CAR_VALUE

    # map_env = map_env.ravel
    # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

    return car_location, feedback, map_env, goal_distance, step, reset, status

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
