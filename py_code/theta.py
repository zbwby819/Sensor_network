# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:35:13 2020

@author: azrael
"""
def theta(env, start_loc, end_loc):           #theta*   return  connection points
    start_loc = (round(start_loc[0]), round(start_loc[-1])) 
    end_loc = (round(end_loc[0]), round(end_loc[-1])) 
    s_path = AStarSearch(env, start_loc, end_loc)
    cur_start = s_path[0]
    record_grid = []
    set_grid = s_path[0]
    record_grid.append(s_path[0])
    for i in range(1, len(s_path)):       # find line
        cur_loc = s_path[i]
        past_grid = []
        past_grid_int = []
        x_flag = 1
        y_flag = 1
        cur_x, cur_y = cur_start[0], cur_start[1]
        if (cur_start[0] - cur_loc[0]) >=0:
            x_flag = -1
        if (cur_start[1] - cur_loc[1]) >=0:
            y_flag = -1
        if abs(cur_loc[0]-cur_start[0])<= abs(cur_loc[1]-cur_start[1]):    
            slocp = abs((cur_loc[0]-cur_start[0])/(cur_loc[1]-cur_start[1]))
            for j in range(abs(cur_start[1]- cur_loc[1])):
                past_cell = (cur_start[0]+ j*slocp*x_flag, cur_start[1]+j*y_flag)
                past_grid.append(past_cell)
                low_bound_x = int(past_cell[0])
                high_bound_x = math.ceil(past_cell[0])
                print('cur_x:',cur_x, '  low_bound_x:',low_bound_x, '  high_bound_x:', high_bound_x)
                if past_cell[0] != int(past_cell[0]): 
                    if low_bound_x == cur_x:                        
                        past_grid_int.append((low_bound_x, cur_start[1]+j*y_flag))
                        past_grid_int.append((high_bound_x, cur_start[1]+j*y_flag))
                        cur_x = high_bound_x
                    elif high_bound_x == cur_x:                        
                        past_grid_int.append((high_bound_x, cur_start[1]+j*y_flag))
                        past_grid_int.append((low_bound_x, cur_start[1]+j*y_flag))
                        cur_x = low_bound_x
                else:
                    cur_x, cur_y = past_cell
        else:  
            slocp = abs((cur_loc[1]-cur_start[1])/(cur_loc[0]-cur_start[0]))
            for j in range(abs(cur_start[0]-cur_loc[0])):
                past_cell = (cur_start[0]+j*x_flag, cur_start[1]+slocp*j*y_flag)
                past_grid.append(past_cell)
                low_bound_y = int(past_cell[1])
                high_bound_y = math.ceil(past_cell[1])
                print('cur_y:',cur_x, '  low_bound_y:',low_bound_x, '  high_bound_y:', high_bound_x)
                if low_bound_y == cur_y:                        
                    past_grid_int.append((cur_start[0]+j*x_flag, low_bound_y))
                    past_grid_int.append((cur_start[0]+j*x_flag, high_bound_y))
                    cur_y = high_bound_y
                elif high_bound_x == cur_x:    
                    past_grid_int.append((cur_start[0]+j*x_flag, high_bound_y))                    
                    past_grid_int.append((cur_start[0]+j*x_flag, low_bound_y))
                    cur_y = low_bound_y               
                
                








    start_loc = (round(start_loc[0]), round(start_loc[-1])) 
    end_loc = (round(end_loc[0]), round(end_loc[-1])) 
    s_path = AStarSearch(env, start_loc, end_loc)
    cur_start = s_path[0]
    record_grid = []
    set_grid = s_path[0]
    record_grid.append(s_path[0])
    for i in range(1, len(s_path)):      
        cur_loc = s_path[i]
        past_grid = []
        x_flag = 1
        y_flag = 1
        if (cur_start[1] - cur_loc[1]) >=0:
            y_flag = -1
        if (cur_start[0] - cur_loc[0]) >=0:
            x_flag = -1
        if abs(cur_loc[0]-cur_start[0])<= abs(cur_loc[1]-cur_start[1]):    
            slocp = abs((cur_loc[0]-cur_start[0])/(cur_loc[1]-cur_start[1]))
            for j in range(abs(cur_start[1]- cur_loc[1])):
                past_grid.append((cur_start[0]+ j*slocp*x_flag, cur_start[1]+j*y_flag))
        else:  
            slocp = abs((cur_loc[1]-cur_start[1])/(cur_loc[0]-cur_start[0]))
            for j in range(abs(cur_start[0]-cur_loc[0])):
                past_grid.append((cur_start[0]+j*x_flag, cur_start[1]+slocp*j*y_flag))
                
        past_grid_int = []
        #past_grid_int.append(past_grid[0])
        past_x = int(past_grid[0][0])
        past_y = int(past_grid[0][1])
        for k in range(len(past_grid[:-1])):         # normal to int
            cur_grid = past_grid[k]
            p_grid = past_grid[k+1]
            if int(past_grid[k][0]) != int(past_grid[k+1][0]):
                past_grid_int.append(int(past_grid[k][0]), int(past_grid[k][1]))
                past_grid_int.append(int(past_grid[k+1][0]), int(past_grid[k+1][1]))
                
            
            if int(past_grid[k][0]) != int(past_grid[k][0]):
                low_bound_x = int(cur_grid[0])
                high_bound_x = math.ceil(cur_grid[0])
                if low_bound_x == past_x:
                    past_y += 1
                    past_grid_int.append((low_bound_x, past_y))
                    past_grid_int.append((high_bound_x, past_y))
                    
                    past_x = high_bound_x
                    print('1')
                    continue
                if high_bound_x == past_x:
                    past_y -= 1
                    past_grid_int.append((high_bound_x, past_y))
                    past_grid_int.append((low_bound_x, past_y
                    
                    past_x = low_bound_x
                    print('2')))
                    continue
            elif cur_grid[1] != int(cur_grid[1]):
                low_bound_y = int(cur_grid[1])
                high_bound_y = math.ceil(cur_grid[1])
                if low_bound_y == past_y:
                    past_x += 1
                    past_grid_int.append((past_x, low_bound_y))
                    past_grid_int.append((past_x, high_bound_y))
                    
                    past_y = high_bound_y
                    print('3')
                    continue
                if high_bound_y == past_y:
                    past_x -= 1
                    past_grid_int.append((past_x, high_bound_y))
                    past_grid_int.append((past_x, low_bound_y))
                    
                    past_y = low_bound_y
                    print('4')
                    continue
            else:
                past_grid_int.append((int(cur_grid[0]), int(cur_grid[1])))
                past_x = int(past_grid[0][0])
                past_y = int(past_grid[0][1])
                
                
                past_grid_int.append((np.int(cur_grid[0]), cur_grid[1]))
                past_grid_int.append((math.ceil(cur_grid[0]), cur_grid[1]))
            elif type(cur_grid[1]) != int:
                past_grid_int.append((cur_grid[0], np.int(cur_grid[1])))
                past_grid_int.append((cur_grid[0], math.ceil(cur_grid[1])))                
        for p_i, cur_grid_int in range(len(past_grid_int)):
            if env[cur_grid_int[0], cur_grid_int[1]] == 0:
                pass
            else:
                record_grid.append(set_grid)
                cur_start = cur_loc   
                break
        set_grid = cur_loc   
    record_grid.append(s_path[-1])
    return record_grid