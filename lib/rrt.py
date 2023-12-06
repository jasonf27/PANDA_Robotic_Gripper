import numpy as np
import random
from lib.detectCollision import detectCollision, detectCollisionOnce
from lib.loadmap import loadmap
from copy import deepcopy
import collections
from lib.calculateFK import FK
from lib.solveIK import IK
import time
from matplotlib.pyplot import plot

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = list()

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])


    #initialize the node graphs as an empty dictionaries
    #tree that starts at the start configuration
    graph_start = collections.defaultdict(list)
    #tree that starts at the goal configuration
    graph_goal = collections.defaultdict(list)


    #check if the robot can move directly to the goal configuration
    if isPathClear(start,goal,map):
        path = path + [start] + [goal]
        #path.append(start) #points between start and goal
        #path.append(goal)
        return path

    i=1
    max_counts = 1e2
    while i <= max_counts:
        connects_to_start = False
        connects_to_goal = False


        #Generate random configuration within joint limits
        qnew = (upperLim - lowerLim) * np.random.random((7,)) + lowerLim

        #if new config collides, generate new configuration
        if isRobotCollided(qnew, map):
            continue

        #check if path to config from start or nearest is valid, if so add to start graph
        qnearest_start = nearest(graph_start, start, array_to_str(qnew))
        if isPathClear(start, qnew, map):
            graph_start[array_to_str(start)].append(array_to_str(qnew))
            connects_to_start = True
        elif isPathClear(qnearest_start, qnew, map):
            graph_start[array_to_str(qnearest_start)].append(array_to_str(qnew))
            connects_to_start = True

        #check if path to config from goal is valid, if so add to goal graph
        qnearest_goal = nearest(graph_goal, goal, array_to_str(qnew))
        if isPathClear(goal, qnew, map):
            graph_goal[array_to_str(goal)].append(array_to_str(qnew))
            connects_to_goal = True
        elif isPathClear(qnearest_goal, qnew, map):
            graph_goal[array_to_str(qnearest_goal)].append(array_to_str(qnew))
            connects_to_goal = True

        #if point doesn't connect to either graph generate new config
        if not (connects_to_start or connects_to_goal):
            continue
        #if point
        if connects_to_start and connects_to_goal:
            path_start = find_path(graph_start, start, qnew, path=[])
            path_goal = find_path(graph_goal, goal, qnew, path=[])
            #print(path_goal)
            #print(list(reversed(path_goal))[1:])
            path = path + path_start + list(reversed(path_goal))[1:]
            break

        i = i + 1

    if i>max_counts:
        path = []
        print("exceeded max counts")

    return path

def isRobotCollided(q, map):
    fk = FK()
    joint_positions, T0e, T0i = fk.forward(q, 2)
    obstacles = map.obstacles

    for x in obstacles:
        for y in range(1,7):
            collisions = detectCollisionOnce(joint_positions[y-1,:], joint_positions[y,:], x)
            if (collisions):
                return collisions
                break

    return False

def isPathClear(q1,q2,map):
    steps = 50
    steps_array = (q2-q1)/steps
    q_test = q1
    for x in range(steps):
        q_test = q_test + steps_array
        collided = isRobotCollided(q_test,map)
        if collided:
            return False
            break
    return True

def nearest(graph, qbase, qnew):
    #print('nearest')
    #print(graph)
    if not bool(graph):
        return qbase

    min_dist = 1e3
    for k, v in graph.items():
        #print(k)
        dist = np.linalg.norm(str_to_array(k)-str_to_array(qnew))
        if dist < min_dist:
            min_dist = dist
            nearest_node = k
    return str_to_array(nearest_node)

def find_path(graph, start, goal, path=[]):
    path = path + [start]
    if np.allclose(start, goal):
        return path
    if array_to_str(start) not in graph:
        return None
    for node in graph[array_to_str(start)]:
        in_path = False
        for x in path:
            if np.allclose(str_to_array(node), x):
                in_path = True
                break
        if not in_path:
            newpath = find_path(graph, str_to_array(node), goal, path)
            if newpath: return newpath
    return None
#used python.org/doc/essays/graphs for the function above

def array_to_str(arg):
    str = np.array2string(arg)
    str = str.replace('[', '')
    str = str.replace(']', '')
    str = str.replace('\n', '')
    return str

def str_to_array(arg):
    return np.fromstring(arg, sep=' ', dtype=float, count=7)

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    for x in path:
        print(x)
