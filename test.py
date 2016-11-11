import numpy as np

def random_map(dim1, dim2, propobility):
  map_matrix = np.zeros([dim1, dim2])
  for i in range(dim1):
    for j in range(dim2):
      a = np.random.random(1)
      if a < propobility:
        map_matrix[i,j] = 1
  start = np.random.random_integers(0, 9, 2)
  start_x = start[0]
  start_y = start[1]
  start = start_x, start_y
  goal = np.random.random_integers(0, 9, 2)
  goal_x = goal[0]
  goal_y = goal[1]
  goal = goal_x, goal_y
  map_matrix[goal] = 2
  return map_matrix, start, goal

dim1 = 10
dim2 = 10
probobility = 0.3
map_matrix, initial_car_location, goal_location = random_map(dim1, dim2, probobility)

print map_matrix[initial_car_location]