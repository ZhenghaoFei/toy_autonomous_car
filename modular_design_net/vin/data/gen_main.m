goal = [6,6];
maze = obstacle_gen(8, goal, 2)

maze.add_rand_obs('rect')
maze.add_N_rand_obs(5)
maze.add_border()
map = maze.getimage()
imshow(map(:,:,1))