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
from partner_lib.calculateFK import FK
from partner_lib.calcJacobian import FK
from partner_lib.solveIK import IK
from partner_lib.rrt import rrt
from partner_lib.loadmap import loadmap

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

def robotGetBlock():
    """
    Implement RRT algorithm in this file.
    :param static transformation tags: list of size 4 where each index is the
    transformation of the block in the robot frame
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    # move the robot arm to the right position and orientation to pick up
        # the block using the inverse function
        # if there isn't a success, move on to the next static block
        # if successful, update the seed
            # actually move the arm to that right position
            # use the exec_gripper_command

    return

def robotPlaceBlock():
    """
    Implement RRT algorithm in this file. Also need to add to the list of the 
    obstacles.
    :param transformation of the block
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    # currently has block, need to move to scoring platform
    # add the block height to the current height of the tower
    # move to that position
    # drop the block
    return

if __name__ == "__main__":

    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    # arm.safe_move_to_position(arm.neutral_position()) # on your mark!

    ik = IK()

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
