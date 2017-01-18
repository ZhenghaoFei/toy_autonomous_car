# From http://code.activestate.com/recipes/578356-random-maze-generator/
# Modified to use numpy and return maze instead of saving it as an image.

# Random Maze Generator using Depth-first Search
# http://en.wikipedia.org/wiki/Maze_generation_algorithm
# FB - 20121214


import random
import numpy as np
import matplotlib.pyplot as plt
# def make_maze(mx, my):
# 	maze = np.zeros((my, mx))
# 	dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
# 	# start the maze from (0, 0)
# 	stack = [(0, 0)]#[(random.randint(0, mx - 1), random.randint(0, my - 1))]

# 	while len(stack) > 0:
# 	    (cx, cy) = stack[-1]
# 	    maze[cy][cx] = 1
# 	    # find a new cell to add
# 	    nlst = [] # list of available neighbors
# 	    for i in range(4):
# 	        nx = cx + dx[i]; ny = cy + dy[i]
# 	        if nx >= 0 and nx < mx and ny >= 0 and ny < my:
# 	            if maze[ny][nx] == 0:
# 	                # of occupied neighbors must be 1
# 	                ctr = 0
# 	                for j in range(4):
# 	                    ex = nx + dx[j]; ey = ny + dy[j]
# 	                    if ex >= 0 and ex < mx and ey >= 0 and ey < my:
# 	                        if maze[ey][ex] == 1: ctr += 1
# 	                if ctr == 1: nlst.append(i)
# 	    # if 1 or more neighbors available then randomly select one and move
# 	    if len(nlst) > 0:
# 	        ir = np.random.choice(nlst)
# 	        cx += dx[ir]; cy += dy[ir]
# 	        stack.append((cx, cy))
# 	    else: stack.pop()

# 	return np.abs(maze.T - 1)  # transpose and invert 0s and 1s
def make_maze(mx,my, WALL_VALUE):
	maze = np.zeros((my, mx))
	dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
	# start the maze from (0, 0)
	stack = [(np.random.randint(0, mx - 1), np.random.randint(0, my - 1))]#[(0, 0)]
	#start = stack[0]
	while len(stack) > 0:
		(cx, cy) = stack[-1]
		maze[cy][cx] = WALL_VALUE
		start = (cy,cx)	
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
	# maze[goal] = 200

	return maze, start, goal  # transpose and invert 0s and 1s
def plot_map(map_matrix, car_location, goal_location):
    map_matrix[car_location] = 100# use three to present car
    map_matrix[goal_location] = 200# use three to present car
    plt.imshow(map_matrix, interpolation='none')
    plt.show()

if __name__ == '__main__':
	map_matrix, car_location, goal_location = make_maze(10, 10, 10)
	plot_map(map_matrix, car_location, goal_location)
