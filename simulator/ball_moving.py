#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import pygame


def createblock(length, height, color):
    tmpblock = pygame.Surface((length, height))
    tmpblock.fill(color)
    tmpblock.convert()
    return tmpblock
# create map on background
def map_create(tilemap,background,lines,columns,length,height):
    wallblock = createblock(length, height, BLACK)

    endblock  = createblock(length, height, RED)
    startblock = createblock(length,height,GREEN)
    for y in range(lines):
        for x in range(columns):
            if tilemap[y][x] == "x": # wall
                background.blit(wallblock, (length * x, height * y))
            elif tilemap[y][x] == "s": # start
                ballx_start = length * x
                bally_start = height * y
                background.blit(startblock,(length * x, height * y))
            elif tilemap[y][x] == "e":
                ballx_end = length * x
                bally_end = height * y
                background.blit(endblock,(length * x, height * y))
    background = background.copy()
    screen.blit(background,(0,0))

    return ballx_start, bally_start, ballx_end, bally_end, lines, columns,background
def collision_check(ballx,bally,tilemap,length,height,lines,columns):
    x_tile = int(ballx/length)
    y_tile = int(bally/height)
    # print(x_tile,y_tile)
    if tilemap[y_tile][x_tile] == 'x' or ballx < 0 or bally < 0 or y_tile > columns or x_tile > lines:
        return 1
    else: 
        return 0

#constants representing colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
# create tile map as you like
tilemap = [ "xxx.xxxxxxxxxxxxxxxxxx",
            "s......x.............x",
            "xxxx.........xx......x",
            "x......x....x.x......x",
            "x......x......x......x",
            "x......x......x......x",
            "x...xxxxxx....x......x",
            "x......x.............x",
            "x......x......xxxxxxxx",
            "xxxxxx.x.............x",
            "x......x.............x",
            "x......x.............x",
            "x..........xxxx...xxxx",
            "x..........x.........x",
            "xxxxxxxxxxxxxxxxxexxxx"]
# start a game window
pygame.init()
screen = pygame.display.set_mode((440,300)) # set the size of screen
screenrect = screen.get_rect() # obtain the height and width of screen
# create background
background = pygame.Surface((screen.get_size()))
background.fill(WHITE) # white background
background = background.convert()
background0 = background.copy()
screen.blit(background,(0,0)) # display the background
# construct a ball to move
ballsurface = pygame.Surface((10,10))
ballsurface.set_colorkey(BLACK)
pygame.draw.circle(ballsurface,RED,(5,5),5)
ballsurface = ballsurface.convert_alpha()
ballrect = ballsurface.get_rect()
print(ballrect.width,ballrect.height)

lines = len(tilemap) # dimension of blocks on x directon
columns = len(tilemap[0]) # dimension of blocks on y direction
length = screenrect.width / columns
height = screenrect.height/ lines
print(length,height)

# create the map
ballx_start, bally_start, ballx_end, bally_end, lines, columns,background = map_create(tilemap,background0,lines,columns,length,height)
# clock = pygame.time.Clock()
done = True

# ballx and bally mark the location of ball
ballx = ballx_start
bally = bally_start
print(ballx,bally)
# up, down, left right boolean
up = 0
down = 0
right = 0
left = 0
# test with keytelecop
while done:
    # millisenconds = clock.ticks()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            # pygame window closed by user
            done = False 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = False 
            if event.key == pygame.K_UP:
                up = 1
                print("uppressed")
            if event.key == pygame.K_DOWN:
                down = 1
                print("downpressed")
            if event.key == pygame.K_RIGHT:
                right = 1
                print("rightpressed")
            if event.key == pygame.K_LEFT:
                left = 1
                print("leftpressed")
    # show the background
    pygame.display.set_caption("Press cursor key to move ball")
    screen.blit(background,(0,0))
    # # ---------check if the ball collision the wall
    if up==1:
        bally -= ballrect.height
        print (ballx, bally)
        up = 0
    if down==1:
        bally += ballrect.height
        down = 0
        print (ballx, bally)
    if left==1:
        ballx -= ballrect.width
        left =0
        print (ballx, bally)
    if right==1:
        ballx += ballrect.width
        right = 0    
        print (ballx, bally)
    coll_bool = collision_check(ballx,bally,tilemap,length,height,lines,columns)
    if coll_bool:
        ballx = ballx_start
        bally = bally_start # back to the starting point
        print("collision!")
    # move point surface
    screen.blit(ballsurface, (ballx, bally))
    pygame.display.flip() # flip the screen 30 times a second

pygame.quit()