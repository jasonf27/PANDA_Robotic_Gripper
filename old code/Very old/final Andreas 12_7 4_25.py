import sys
import numpy as np
from copy import deepcopy

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

from testing import transform


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

def getTags(team, tags):
    """
    Returns the transformation of the tags in the robot frame.
    :param
        team: string "red" or "blue"
        tags: list of tuples (tag_name, pose) size 10 where tag_name is the tag
        labeled from 0 to 12 where tag 0 is the static tag, tags 1-6 are faces
        on the static block, and tags 7-12 are faces on the dynamic blocks
        tag0 will be included along with the 4 tags corresponding to red or blue
        static team's blocks depending on what team we are assigned
    :return:
        Trobot_tag_list: list of tuples (tag_name, pose) size 4
        where each index is the transformation of a tag on a static block in
        the robot frame
    """
    static_tag_names = {"tag1", "tag2", "tag3", "tag4", "tag5", "tag6"}

    # constants
    tag0_x_distance = -0.5

    Trobot_tag0 = np.array([
        [1, 0, 0, tag0_x_distance],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    Trobot_tag_list = []

    for (name, pose) in tags:
        if name == "tag0":
            # permanent tag
            Tcamera_tag0 = pose
            Ttag0_camera = np.linalg.inv(Tcamera_tag0)
        elif name in static_tag_names:
            # Trobot_tag0 doesn't need to be added to the list of blocks
            # fetching dynamic blocks blindly, so don't need to save those
            # transformations
            Ttag0_tagi = np.matmul(Ttag0_camera, pose)
            Trobot_tagi = np.matmul(Trobot_tag0, Ttag0_tagi)

            Trobot_tag_list.append((name, Trobot_tagi))

    return Trobot_tag_list

def getBlocks():
    """
    Returns the block transformations using the transformations of the tags in
    the robot frame.
    :param
        Trobot_tag_list: list of tuples (tag_name, pose) size 4
        where each index is the transformation of a tag on a static block in
        the robot frame
    :return:
        Trobot_block_list: list of size 4 where each index is the
        4x4 transformation of the static block in the robot frame
    """
    Tblock_tag_dict = {
        "tag1": np.array([
            [0, 0, 1, 0.025],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]),
        "tag2": np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0.025],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]),
        "tag3": np.array([
            [0, 0, -1, -0.025],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]),
        "tag4": np.array([
            [1, 0, 0, 0],
            [0, 0, -1, -0.025],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]),
        "tag5": np.array([
            [0, 0, -1, -0.025],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]),
        "tag6": np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0.025],
            [0, 0, 0, 1]
        ]),
    }

    Trobot_block_list = []

    for (name, pose) in Trobot_tag_list:
        # Trobot_block = Trobot_tag*Ttag_block
        # Ttag_block = inv(Tblock_tag) where Tblock_tag is precalculated and
        # always stays the same
        Tblock_tag = Tblock_tag_dict[name]
        Ttag_blcok = np.linalg.inv(Tblock_tag)
        Trobot_block = np.matmul(pose, Tblock_tag)
        Trobot_block_list.append(Trobot_block)
    return Trobot_block_list

def getListOfPickUpTransformations():
    """
    For a single block, get a list of posible transformations that the robot
    can go to in order to pick up the block.
    :param static transformation tags: list of size 4 where each index is the
    transformation of the block in the robot frame
    :return:
        a list of 4x4 transformations that the robot can go to to pick up the blocks
        a list of 4x4 transformations from robot EE to block frame
    """
    return


def goToBlock(listOfPoses, map, arm):
    #convert listOfPoses into arm configs
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    listOfConfigs = list()
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

def robotPlaceBlock(team, map, arm, height):
    """
    Implement RRT algorithm in this file. Also need to add to the list of the
    obstacles.
    :param transformation of the block
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    # currently has block, need to move to scoring platform
    # add the block height to the current height of the tower
    #maybe add logic if block needs to be rotated to get white side on top
    #platform [x,y,z] = [+-0.809, 0.562, 0.200] (+ for blue, - for red)
    if (team == "blue"):
        disp = np.array([0.562, -0.978+0.809, 0.200+0.030+0.05*height])
    else:
        disp = np.array([0.562, 0.978-0.809, 0.200+0.030+0.05*height])

    #transformation from end effector to world frame
    #places block down vertically
    target = transform(disp, np.array([pi, 0, 0]) )

    #need seed
    seed = np.array([0.26228134, 0.22685618, 0.03102079, -2.03994858, -0.00908943, 2.266679, 1.08373297])
    q, success, rollout = ik.inverse(H, seed)

    moveToPose(arm.get_positions(), q, map,arm)
    print("Block Placed")

    # drop the block, 0.075 for 75 mm, 0 for 0 N
    exec_gripper_command(0.075, 0)

    return
'''
def fuckShitUp():
    #messes up blocks on dynamic turntable
    danger_pose =
    moveToPose(arm.get_positions(), danger_pose, map, arm)

    return
'''

def rotateBlock(team, height):
    #function assumes IK solver will always be successfull
    if (team=="red"):
        seed1 = [-0.1043823, 0.15690434, 0.31329159, -2.26284303, -1.84511704, 2.95556629, 2.74703923]
        seed2 = [0.22851529, 0.7711638, -0.00443569, -1.17437424, 0.17425965, 1.17801406, 1.01118416]
        target1 = transform(np.array([0.562, 0.978-0.809, 0.200+0.030+0.05*height]), np.array([pi, -pi/4, 0]) )
        target2 = transform(np.array([0.562, 0.978-0.809, 0.200+0.030+0.05*height]), np.array([pi, pi/4, 0]) )
    else:
        #haven't done blue yet
    q1, success, rollout = ik.inverse(target1, seed1)
    q2, success, rollout = ik.inverse(target2, seed2)

    exec_gripper_command(0.1, 0)
    arm.safe_move_to_position(q1)
    exec_gripper_command(0.05)
    arm.safe_move_to_position(q2)

    #eventually can create a library of robot poses instead of using iksolver

    return


if __name__ == "__main__":

    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    ik = IK()
    arm = ArmController()
    detector = ObjectDetector()

    # arm.safe_move_to_position(arm.neutral_position()) # on your mark!


    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE
    # get the static tags
    Trobot_tag_list = getTags(team, detector.get_detections())

    # get a list of the transformations of the static blocks in the robot frame
    Trobot_block_list = getBlocks(Trobot_tag_list)

    # create a new list of transformations of where the end effector needs to
    # be in order to pick up the static blocks

    # while the end effector transformations isn't empty and haven't tried more
    # than x (maybe 8) times
        # get the next transformation
        # use robot get block function
        # if the robot successfull got the block, use robotPlaceBlock

    height = 0
    for x in Trobot_block_list:
        goToBlock()
        robotPlaceBlock(team, map, arm, height)
        height = height + 1



    # Tried to stack all of the static blocks
    # Spend the rest of the time stacking the dynamic blocks
    # while True:
    # place the robot arm in the configuration so that the gripper is tangent
    # to the rotating platform
    # wait a specified amount of time
    # try to grasp
    # after a couple uncessful attempts, update the configuration so that
    # the gripper is tangent to the platform but closer to the platform's center

    # Detect some tags...
    # for (name, pose) in detector.get_detections():
    #      print(name,'\n',pose)

    # END STUDENT CODE
    #test configs
    testConfigs(team)
