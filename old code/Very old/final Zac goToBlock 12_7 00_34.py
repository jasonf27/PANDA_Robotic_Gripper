import sys

sys.path.append("/Users/appollo_liu/Documents/workspace/school/meam-520/meam520_labs/")

import numpy as np
from copy import deepcopy
from math import pi, acos

import rospy
import roslib

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# The library you implemented over the course of this semester!
from lib.calculateFK import FK
from lib.calcJacobian import FK
from lib.solveIK import IK
from lib.rrt import rrt
from lib.loadmap import loadmap

def moveToPose(current, target, map,arm):
    path = rrt(map,current,target)
    print("RRT Run")
    if path == []:
        return 1
    else:
        for x in path:
            arm.safe_move_to_position(x)
            print(x)
        return 0


def goToBlock(listOfPoses, map, arm):
    #convert listOfPoses into arm configs
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    listOfConfigs = list()
    ik = IK()
    for x in listOfPoses:
        print(x)
        qnew, success, rollout = ik.inverse(x,q)
        print(qnew)
        print(success)
        if success:
            listOfConfigs.append(qnew)


    #Open up gripper before any movement
    arm.exec_gripper_cmd(0.5,0.1) #DOUBLE CHECK PARAMETERS OF FUNCTION
    print("Gripper Opened")
    #get current configuration of arm
    current = arm.get_positions()
    print("Current Position Found")
    #loop through list of valid poses, move to first valid pose
    print(listOfConfigs)
    index = 0
    for x in listOfConfigs:
        result = moveToPose(current,x,map,arm)
        if result == 0:
            success = True
            break
        if result == 1:
            success = False
            print("INVALID CONFIG")
        index = index + 1
    if not success:
        index = -1
    #close gripper
    arm.exec_gripper_cmd(0,0.5)
    return index,success



if __name__ == "__main__":

    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    arm.safe_move_to_position(arm.neutral_position()) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    # Detect some tags...
    #tags = detector.get_detections()
    #Tcamera_tag0 = tags["tag0"]

    #transformations = []

    # get all transformations of the blocks in the robot frame
    #for (name, pose) in tags:
    #     print(name,'\n',pose)

    #


    # Move around...

    listOfPoses = list()

    listOfPoses.append(np.array([
        [0,-1,0,0.3],
        [-1,0,0,0],
        [0,0,-1,.5],
        [0,0,0, 1],
    ]))

    listOfPoses.append(np.array([
        [0,-1,0,0.3],
        [-1,0,0,0],
        [0,0,-1,0],
        [0,0,0, 1],
    ]))


    map_struct = loadmap("../../maps/map1.txt")
    index, success = goToBlock(listOfPoses,deepcopy(map_struct),arm)
    print(index)
    print(success)

    # END STUDENT CODE
