# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:45:31 2020

@author: azrael
"""
#############################################################  随机Prim
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
 
num_rows = 4 # number of rows
num_cols = 4 # number of columns
 
# The array M is going to hold the array information for each cell.
# The first four coordinates tell if walls exist on those sides 
# and the fifth indicates if the cell has been visited in the search.
# M(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)
M = np.zeros((num_rows,num_cols,5), dtype=np.uint8)
 
# The array image is going to be the output image to display
image = np.zeros((num_rows*10,num_cols*10), dtype=np.uint8)
 
# Set starting row and column
r = 0
c = 0
history = [(r,c)] # The history is the stack of visited locations
 
# Trace a path though the cells of the maze and open walls along the path.
# We do this with a while loop, repeating the loop until there is no history, 
# which would mean we backtracked to the initial start.
while history: 
	#random choose a candidata cell from the cell set histroy
	r,c = random.choice(history)
	M[r,c,4] = 1 # designate this location as visited
	history.remove((r,c))
	check = []
	# If the randomly chosen cell has multiple edges 
    # that connect it to the existing maze, 
	if c > 0:
		if M[r,c-1,4] == 1:
			check.append('L')
		elif M[r,c-1,4] == 0:
			history.append((r,c-1))
			M[r,c-1,4] = 2
	if r > 0:
		if M[r-1,c,4] == 1: 
			check.append('U') 
		elif M[r-1,c,4] == 0:
			history.append((r-1,c))
			M[r-1,c,4] = 2
	if c < num_cols-1:
		if M[r,c+1,4] == 1: 
			check.append('R')
		elif M[r,c+1,4] == 0:
			history.append((r,c+1))
			M[r,c+1,4] = 2 
	if r < num_rows-1:
		if M[r+1,c,4] == 1: 
			check.append('D') 
		elif  M[r+1,c,4] == 0:
			history.append((r+1,c))
			M[r+1,c,4] = 2
 
    # select one of these edges at random.
	if len(check):
		move_direction = random.choice(check)
		if move_direction == 'L':
			M[r,c,0] = 1
			c = c-1
			M[r,c,2] = 1
		if move_direction == 'U':
			M[r,c,1] = 1
			r = r-1
			M[r,c,3] = 1
		if move_direction == 'R':
			M[r,c,2] = 1
			c = c+1
			M[r,c,0] = 1
		if move_direction == 'D':
			M[r,c,3] = 1
			r = r+1
			M[r,c,1] = 1
         
# Open the walls at the start and finish
#M[0,0,0] = 1
#M[num_rows-1,num_cols-1,2] = 1
    
edge_1, edge_2 = 1,9
# Generate the image for display
for row in range(0,num_rows):
    for col in range(0,num_cols):
        cell_data = M[row,col]
        for i in range(10*row+edge_1,10*row+edge_2):
            image[i,range(10*col+edge_1,10*col+edge_2)] = 255
        if cell_data[0] == 1: 
            image[range(10*row+edge_1,10*row+edge_2),10*col] = 255
            image[range(10*row+edge_1,10*row+edge_2),10*col+1] = 255
        if cell_data[1] == 1: 
            image[10*row,range(10*col+edge_1,10*col+edge_2)] = 255
            image[10*row+1,range(10*col+edge_1,10*col+edge_2)] = 255
        if cell_data[2] == 1: 
            image[range(10*row+edge_1,10*row+edge_2),10*col+9] = 255
            image[range(10*row+edge_1,10*row+edge_2),10*col+8] = 255
        if cell_data[3] == 1: 
            image[10*row+9,range(10*col+edge_1,10*col+edge_2)] = 255
            image[10*row+8,range(10*col+edge_1,10*col+edge_2)] = 255
        
 
# Display the image
plt.imshow(image, cmap = cm.Greys_r, interpolation='none')
plt.show()
########################################################################### 深度优先（递归回溯）

num_rows = 4 # number of rows
num_cols = 4 # number of columns

M = np.zeros((num_rows,num_cols,5), dtype=np.uint8)
image = np.zeros((num_rows*10,num_cols*10), dtype=np.uint8)
r = 0
c = 0
history = [(r,c)]
while history: 
    M[r,c,4] = 1 # designate this location as visited
    # check if the adjacent cells are valid for moving to
    check = []
    if c > 0 and M[r,c-1,4] == 0:
        check.append('L')  
    if r > 0 and M[r-1,c,4] == 0:
        check.append('U')
    if c < num_cols-1 and M[r,c+1,4] == 0:
        check.append('R')
    if r < num_rows-1 and M[r+1,c,4] == 0:
        check.append('D')    
    
    if len(check): # If there is a valid cell to move to.
        # Mark the walls between cells as open if we move
        history.append([r,c])
        move_direction = random.choice(check)
        if move_direction == 'L':
            M[r,c,0] = 1
            c = c-1
            M[r,c,2] = 1
        if move_direction == 'U':
            M[r,c,1] = 1
            r = r-1
            M[r,c,3] = 1
        if move_direction == 'R':
            M[r,c,2] = 1
            c = c+1
            M[r,c,0] = 1
        if move_direction == 'D':
            M[r,c,3] = 1
            r = r+1
            M[r,c,1] = 1
    else: # If there are no valid cells to move to.
    # retrace one step back in history if no move is possible
        r,c = history.pop()

edge_1, edge_2 = 1,9
# Generate the image for display
for row in range(0,num_rows):
    for col in range(0,num_cols):
        cell_data = M[row,col]
        for i in range(10*row+edge_1,10*row+edge_2):
            image[i,range(10*col+edge_1,10*col+edge_2)] = 255
        if cell_data[0] == 1: 
            image[range(10*row+edge_1,10*row+edge_2),10*col] = 255
            image[range(10*row+edge_1,10*row+edge_2),10*col+1] = 255
        if cell_data[1] == 1: 
            image[10*row,range(10*col+edge_1,10*col+edge_2)] = 255
            image[10*row+1,range(10*col+edge_1,10*col+edge_2)] = 255
        if cell_data[2] == 1: 
            image[range(10*row+edge_1,10*row+edge_2),10*col+9] = 255
            image[range(10*row+edge_1,10*row+edge_2),10*col+8] = 255
        if cell_data[3] == 1: 
            image[10*row+9,range(10*col+edge_1,10*col+edge_2)] = 255
            image[10*row+8,range(10*col+edge_1,10*col+edge_2)] = 255
        
 
# Display the image
plt.imshow(image, cmap = cm.Greys_r, interpolation='none')
plt.show()

##################################################################  递归分割

def Recursive_division(r1, r2, c1, c2, M, image):
	if r1 < r2 and c1 < c2:
		rm = random.randint(r1, r2-1)
		cm = random.randint(c1, c2-1)
		cd1 = random.randint(c1,cm)
		cd2 = random.randint(cm+1,c2)
		rd1 = random.randint(r1,rm)
		rd2 = random.randint(rm+1,r2)
		d = random.randint(1,4)
		if d == 1:
			M[rd2, cm, 2] = 1
			M[rd2, cm+1, 0] = 1
			M[rm, cd1, 3] = 1
			M[rm+1, cd1, 1] = 1
			M[rm, cd2, 3] = 1
			M[rm+1, cd2, 1] = 1
		elif d == 2:
			M[rd1, cm, 2] = 1
			M[rd1, cm+1, 0] = 1
			M[rm, cd1, 3] = 1
			M[rm+1, cd1, 1] = 1
			M[rm, cd2, 3] = 1
			M[rm+1, cd2, 1] = 1
		elif d == 3:
			M[rd1, cm, 2] = 1
			M[rd1, cm+1, 0] = 1
			M[rd2, cm, 2] = 1
			M[rd2, cm+1, 0] = 1
			M[rm, cd2, 3] = 1
			M[rm+1, cd2, 1] = 1
		elif d == 4:
			M[rd1, cm, 2] = 1
			M[rd1, cm+1, 0] = 1
			M[rd2, cm, 2] = 1
			M[rd2, cm+1, 0] = 1
			M[rm, cd1, 3] = 1
			M[rm+1, cd1, 1] = 1
 
		Recursive_division(r1, rm, c1, cm, M, image)
		Recursive_division(r1, rm, cm+1, c2, M, image)
		Recursive_division(rm+1, r2, cm+1, c2, M, image)
		Recursive_division(rm+1, r2, c1, cm, M, image)
 
	elif r1 < r2:
		rm = random.randint(r1, r2-1)
		M[rm,c1,3] = 1
		M[rm+1,c1,1] = 1
		Recursive_division(r1, rm, c1, c1, M, image)
		Recursive_division(rm+1, r2, c1, c1, M, image)
	elif c1 < c2:
		cm = random.randint(c1,c2-1)
		M[r1,cm,2] = 1
		M[r1,cm+1,0] = 1
		Recursive_division(r1, r1, c1, cm, M, image)
		Recursive_division(r1, r1, cm+1, c2, M, image)


num_rows = 4 # number of rows
num_cols = 4 # number of columns
r1 = 0
r2 = num_rows-1
c1 = 0
c2 = num_cols-1
 
M = np.zeros((num_rows,num_cols,5), dtype=np.uint8)
image = np.zeros((num_rows*10,num_cols*10), dtype=np.uint8)
 
Recursive_division(r1, r2, c1, c2, M, image) 

edge_1, edge_2 = 1,9
# Generate the image for display
for row in range(0,num_rows):
    for col in range(0,num_cols):
        cell_data = M[row,col]
        for i in range(10*row+edge_1,10*row+edge_2):
            image[i,range(10*col+edge_1,10*col+edge_2)] = 255
        if cell_data[0] == 1: 
            image[range(10*row+edge_1,10*row+edge_2),10*col] = 255
            image[range(10*row+edge_1,10*row+edge_2),10*col+1] = 255
        if cell_data[1] == 1: 
            image[10*row,range(10*col+edge_1,10*col+edge_2)] = 255
            image[10*row+1,range(10*col+edge_1,10*col+edge_2)] = 255
        if cell_data[2] == 1: 
            image[range(10*row+edge_1,10*row+edge_2),10*col+9] = 255
            image[range(10*row+edge_1,10*row+edge_2),10*col+8] = 255
        if cell_data[3] == 1: 
            image[10*row+9,range(10*col+edge_1,10*col+edge_2)] = 255
            image[10*row+8,range(10*col+edge_1,10*col+edge_2)] = 255
        
 
# Display the image
plt.imshow(image, cmap = cm.Greys_r, interpolation='none')
plt.show()