import sys
import numpy as np
from copy import deepcopy
from math import pi, cos, sin

# THESE IMPORTS SHOULD EXIST FOR ROS SIMULATION / ROBOT TESTING, BUT NOT FOR LOCAL PRINT STATEMENT TESTING
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
from lib.rrt import rrt, isPathClear
from lib.loadmap import loadmap

pose_dict = {
    "blue1": [-0.11059261, 0.23072092, -0.18887226, -2.03966135, 0.05592319, 2.26565009, 0.45499628],
    "bluerot1a": [0.04604489, 0.1546081, -0.25305249, -2.26136344, 1.845263, 2.9635384, -1.18200928],
    "bluerot1b": [-0.09791986, 0.77925483, -0.19677689, -1.17743368, -0.07393123, 1.18825925, 0.49116233],
    "red1": [0.26228134, 0.22685618, 0.03102079, -2.03994858, -0.00908943, 2.266679, 1.08373297],
    "redrot1a": [-0.1043823, 0.15690434, 0.31329159, -2.26284303, -1.84511704, 2.95556629, 2.74703923],
    "redrot1b": [0.22851529, 0.7711638, -0.00443569, -1.17437424, 0.17425965, 1.17801406, 1.01118416],
    "redoff1": [-1.38977994, -0.4156571, 2.2180165, -2.26882491, 1.94831861, 3.45266584, -0.29810282],
    "redoff2": [ 0.34229593, 0.52931882, 0.08015646, -1.58594171, 0.2613949, 1.38592692, 1.16211813],
    "redoffneutral": [ 0.11748554, -0.10904219, 0.22994372, -2.32347635, 0.03107423, 2.21705025, 1.11279223],
    "blueoff1": [-0.25430232, -0.00049506, -0.25403467, -2.45026471, 1.64209212, 2.78940178, -1.23673548],
    "blueoff2": [ 0.11748554, -0.10904219, 0.22994372, -2.32347635, 0.03107423, 2.21705025, 1.11279223],
    "blueoffneutral": [-0.19760394, -0.10738558, -0.15096528, -2.32353152, -0.02019504, 2.21726003, 0.44985139]
    }

def moveToPose(target,arm):
    arm.safe_move_to_position(np.array([-0.09129,  0.09964, -0.21542, -1.50576,  0.02128,  1.6031,   2.04984]))
    print("In Neutral Position Above Static Table")
    #if path == []:
    #    return 1
    #else:
    #    for x in path:
    #        arm.safe_move_to_position(x)
    #        print(x)
    #    return 0
    arm.safe_move_to_position(target)
    print("At Block")


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

    for (name, pose) in tags:
        if name in static_tag_names:
            # Trobot_tag0 doesn't need to be added to the list of blocks
            # fetching dynamic blocks blindly, so don't need to save those
            # transformations
            Ttag0_tagi = np.matmul(Ttag0_camera, pose)
            Trobot_tagi = np.matmul(Trobot_tag0, Ttag0_tagi)

            Trobot_tag_list.append((name, Trobot_tagi))

    return Trobot_tag_list

def getBlocks(Trobot_tag_list):
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
        # "tag5": np.array([  # BEFORE MODIFYING RIGHTMOST d VECTOR
        #     [0, 0, -1, -0.025],
        #     [-1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ]),
        "tag5": np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, -0.025],
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
        Trobot_tag = pose
        Tblock_tag = Tblock_tag_dict[name]
        Ttag_block = np.linalg.inv(Tblock_tag)
        # Trobot_block = np.matmul(Trobot_tag, Tblock_tag) # BEFORE INVERTING Tblock_tag
        Trobot_block = np.matmul(Trobot_tag, Ttag_block) # AFTER INVERTING Tblock_tag
        Trobot_block_list.append(Trobot_block)
    return Trobot_block_list

def getOrderedBlocks(Trobot_block_list):
    """
    Returns the ordered block transformations from closest to farthest from the robot
    base.
    :param
        Trobot_block_list: list of size 4 where each index is the
        4x4 transformation of the sorted(sides_list)tatic block in the robot frame
    :return:
        Trobot_block_ordered_list: list of size 4 where each index is the
        4x4 transformation of the static block closest to the robot base
    """
    # create a list of tuples of (block_distance, Trobot_block)
    block_distance_list = []

    for Trobot_block in Trobot_block_list:
        block_position = Trobot_block[0:3, 3]
        block_distance = np.linalg.norm(block_position)
        block_distance_list.append((block_distance, Trobot_block))

    # print("block_distance_list = ", block_distance_list)

    # use the sorted function
    Trobot_block_ordered_list = [x for _,x in sorted(block_distance_list)]

    return Trobot_block_ordered_list

# Helper: Round a 3D vector to the nearest unit vector ex. (0, 0, 1)
def roundVect(vect):
    for i in range(3):
        if vect[i] > 0.9:
            vect[i] = 1
        elif vect[i] < -0.9:
            vect[i] = -1
        else:
            vect[i] = 0
    return vect

# Helper: Orders block sides in ascending order of distance to gripper base
# def orderSides(sidePose_list, Rrobot_block):
#     sides_list = []
#     for sidePose in sidePose_list:
#         sidePosition_BlockCoord = 0.025 * sidePose
#         sidePosition_RobotCoord = Rrobot_block * sidePosition_BlockCoord
#         side_distance = np.linalg.norm(sidePosition_RobotCoord)
#         sides_list.append((side_distance,sidePose,sidePosition_BlockCoord))
#     sides_list_ordered = [(x,y) for (a,x,y) in sorted(sides_list)] # Contains block frame post and position
#     return sides_list_ordered

def getListOfPickUpTransformations(Trobot_block):
    """
    For a single block, get a list of posible transformations that the robot
    can go to in order to pick up the block.
    :param static transformation tags: list of size 4 where each index is the
    transformation of the block in the robot frame
    :return:
        Trobot_EE_list: a list of 4x4 transformations that the robot can go to to pick up the blocks
        Tblock_EE_list: a list of 4x4 transformations from robot EE to block frame
    """

    # Initialize all lists, including intermediary rotation (not yet including translation)
    Rblock_EE_list = []
    Trobot_EE_list = []
    Tblock_EE_list = []

    # Compute more rotation and homogenous transformation matrices
    Rrobot_block = Trobot_block[0:3,0:3]
    Rblock_robot = np.linalg.inv(Rrobot_block)
    Tblock_robot = np.linalg.inv(Trobot_block)

    z = np.array([0, 0, 1]) # Write z vector for future convenience

    # Compute world +z axis pose in block coordinates, round away sensor error
    # Should return a single unit vector, one coord = 1, two coord = 0, block coordinates
    worldZ_BlockCoord = np.matmul(Rblock_robot, z)
    worldZ_BlockCoord = roundVect(worldZ_BlockCoord)

    # CASE 1: HORIZONTAL WHITE SIDE
    if worldZ_BlockCoord[2] == 0:

        c = -np.cross(worldZ_BlockCoord, z)

        # First two valid solutions, just as good
        Rblock_EE_1 = np.transpose(np.array([z, c, -worldZ_BlockCoord]))
        Rblock_EE_list.append(Rblock_EE_1)
        Rblock_EE_2 = np.transpose(np.array([-z, -c, -worldZ_BlockCoord]))
        Rblock_EE_list.append(Rblock_EE_2)

        # Next two valid solutions, worse b/c cannot achieve Side Bonus
        Rblock_EE_3 = np.transpose(np.array([c, -z, -worldZ_BlockCoord]))
        Rblock_EE_list.append(Rblock_EE_3)
        Rblock_EE_4 = np.transpose(np.array([-c, z, -worldZ_BlockCoord]))
        Rblock_EE_list.append(Rblock_EE_4)

        # Note that checking side grips would be a poor usage of time, since top grip is preferable
        # And side grip cannot possibly achieve SB, in pickup and dropoff, given side white
        # With extra time though, I suppose I could add side grip options to the list

    # CASE 2: VERTICAL WHITE SIDE I.E. TOP OR BOTTOM
    else:

        # First four valid solutions, overhead grabbing
        Rblock_EE_1 = np.array([np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1])])
        Rblock_EE_list.append(Rblock_EE_1)
        Rblock_EE_2 = np.array([np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, -1])])
        Rblock_EE_list.append(Rblock_EE_2)
        Rblock_EE_3 = np.array([np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])])
        Rblock_EE_list.append(Rblock_EE_3)
        Rblock_EE_4 = np.array([np.array([0, -1, 0]), np.array([-1, 0, 0]), np.array([0, 0, -1])])
        Rblock_EE_list.append(Rblock_EE_4)

        # Record position and pose of 4 sides, and sort based on proximity to robot base
        # sidePose_list = [np.array([1, 0, 0]),np.array([-1, 0, 0]),np.array([0, 1, 0]),np.array([0, -1, 0])]
        # sides_list_ordered = orderSides(sidePose_list, Rrobot_block)

        # for i in range(2):
        #     sidePose,sidePosition = sides_list_ordered[i]
        #     zBlock_EE = -sidePose

        #     c = np.cross(zBlock_EE, z)

        #     # Two valid solutions
        #     Rblock_EE_i_1 = np.transpose(np.array([z, c, zBlock_EE]))
        #     Rblock_EE_list.append(Rblock_EE_i_1)
        #     Rblock_EE_i_2 = np.transpose(np.array([-z, -c, zBlock_EE]))
        #     Rblock_EE_list.append(Rblock_EE_i_2)

        # Note: For blocks with white side starting face down, may need to flip it into intermediary state
        # to allow for feasible path for gripping and for dropping into stack

        # Note: I stopped after returning 4 side grip transformations, b/c one of them will probably work
        # But I could have continued to return some top grip transformations, which are safer although
        # And on second thought could actually get the Side Bonus b/c simply perform top release into stack
        # Bottom white couldn't work though bc can't release from underneath

    d = 0 # Distance from block to EE, along +zEE axis TO-DO check this!
    dBlock_EE_EECoord = d * np.array([0, 0, 1]) # Frame translation from EE to block, in EE coordinates

    for Rblock_EE in Rblock_EE_list:
        dBlock_EE_BlockCoord = np.matmul(Rblock_EE, dBlock_EE_EECoord) # Convert translation to block coord
        # print("Rblock_EE = ", Rblock_EE)
        # print("dBlock_EE_BlockCoord = ", dBlock_EE_BlockCoord)
        Tblock_EE = np.zeros((4, 4))
        Tblock_EE[0:3, 0:3] = Rblock_EE
        Tblock_EE[0:3, 3] = dBlock_EE_BlockCoord
        Tblock_EE[3, 0:4] = np.array([0, 0, 0, 1])
        # hStack = np.hstack((Rblock_EE, np.transpose(dBlock_EE_BlockCoord)))
        # Tblock_EE = np.vstack((hStack, np.array([0, 0, 0, 1])))
        Tblock_EE_list.append(Tblock_EE)
        Trobot_EE = np.matmul(Trobot_block, Tblock_EE)
        Trobot_EE_list.append(Trobot_EE)

    return Trobot_EE_list, Tblock_EE_list

def pickUpBlock(listOfPoses, map, arm):
    #convert listOfPoses into arm configs
    q = np.array([-0.09129,  0.09964, -0.21542, -1.50576,  0.02128,  1.6031,   2.04984])
    listOfConfigs = list()
    for x in listOfPoses:
        print(x)
        qnew, success, rollout = ik.inverse(x,q)
        print(qnew)
        print(success)
        if success:
            listOfConfigs.append(qnew)
            config = qnew
            xway = x + np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0 ,0 ,0, 0.04], [0, 0, 0, 0]])
            config_way, success_way, rollout = ik.inverse(xway, config)
            if success_way:
                break


    #Open up gripper before any movement
    arm.exec_gripper_cmd(0.75) #DOUBLE CHECK PARAMETERS OF FUNCTION
    print("Gripper Opened")
    #get current configuration of arm
    current = arm.get_positions()
    print("Current Position Found")
    #loop through list of valid poses, move to first valid pose
    #print(listOfConfigs)
    #index = 0
    print("waypoint")
    moveToPose(config_way,arm)
    print("block pose")
    arm.safe_move_to_position(config)
    #close gripper
    arm.exec_gripper_cmd(0)
    print("Gripper Closed")

    arm.safe_move_to_position(np.array([-0.09129,  0.09964, -0.21542, -1.50576,  0.02128,  1.6031,   2.04984]))
    print("Moved to Above Static Table")
    return success

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
    print("placing block")

    if (team == "blue"):
        disp = np.array([0.562, -0.978+0.809, 0.200+0.030+0.05*height])
        seed = pose_dict["blue1"]
    else:
        disp = np.array([0.562, 0.978-0.809, 0.200+0.030+0.05*height])
        seed = pose_dict["red1"]

    #transformation from end effector to world frame
    #places block down vertically
    target = transform(disp, np.array([pi, 0, 0]) )

    #need seed
    #seed = np.array([0.26228134, 0.22685618, 0.03102079, -2.03994858, -0.00908943, 2.266679, 1.08373297])
    q, success, rollout = ik.inverse(target, seed)

    moveToPose(q,arm)
    print("Block Placed")

    # drop the block, 0.075 for 75 mm, 0 for 0 N
    arm.exec_gripper_cmd(0.075, 0)
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
    print("Rotation block at height", height)
    if (team=="red"):
        seed1 = [-0.1043823, 0.15690434, 0.31329159, -2.26284303, -1.84511704, 2.95556629, 2.74703923]
        seed2 = [0.22851529, 0.7711638, -0.00443569, -1.17437424, 0.17425965, 1.17801406, 1.01118416]
        target1 = transform(np.array([0.562, 0.978-0.809, 0.200+0.030+0.05*height]), np.array([pi, -pi/4, 0]) )
        target2 = transform(np.array([0.562, 0.978-0.809, 0.200+0.030+0.05*height]), np.array([pi, pi/4, 0]) )
        mid = transform(np.array([0.562, 0.978-0.809, 0.200+0.080+0.05*height]), np.array([pi, 0, 0]) )
        seedmid = pose_dict["red1"]
    else:
        seed1 = [0.04604489, 0.1546081, -0.25305249, -2.26136344, 1.845263, 2.9635384, -1.18200928]
        seed2 = [-0.09791986, 0.77925483, -0.19677689, -1.17743368, -0.07393123, 1.18825925, 0.49116233]
        target1 = transform(np.array([0.562, -0.978+0.809, 0.200+0.030+0.05*height]), np.array([pi, -pi/4, 0]) )
        target2 = transform(np.array([0.562, -0.978+0.809, 0.200+0.030+0.05*height]), np.array([pi, pi/4, 0]) )
        mid = transform(np.array([0.562, -0.978+0.809, 0.200+0.080+0.05*height]), np.array([pi, 0, 0]) )
        seedmid = pose_dict["blue1"]

    clearance = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0 ,0 ,0, 0.05], [0, 0, 0, 0]])
    target1a = target1 + clearance
    target2a = target2 + clearance

    q1, success, rollout = ik.inverse(target1, seed1)
    print(success)
    q2, success, rollout = ik.inverse(target2, seed2)
    print(success)

    q1a, success, rollout = ik.inverse(target1a, seed1)
    print(success)
    q2a, success, rollout = ik.inverse(target2a, seed2)
    print(success)

    qmid, success, rollout = ik.inverse(mid, seedmid)
    print(success)

    print("Executing")
    #open
    arm.exec_gripper_cmd(0.1, 0)
    print("open")
    #move to -45
    arm.safe_move_to_position(qmid)
    arm.safe_move_to_position(q1a)
    arm.safe_move_to_position(q1)
    #grab
    arm.exec_gripper_cmd(0)
    print("closed")
    #move up .5
    arm.safe_move_to_position(q1a)
    #rotate
    arm.safe_move_to_position(q2a)
    #move down .5
    arm.safe_move_to_position(q2)
    #release
    arm.exec_gripper_cmd(0.1, 0)
    arm.safe_move_to_position(q2a)
    print("done rotating")
    #eventually can create a library of robot poses instead of using iksolver

    return

def rotateOffStack(team):
    #function assumes IK solver will always be successfull
    print("Rotation block off the stack")
    if (team=="red"):
        seed1 = pose_dict["redoff1"]
        seed2 = pose_dict["redoff2"]
        target1 = transform(np.array([0.562-0.1, 0.978-0.809+0.1, 0.200+0.030+0.05*height]), np.array([pi, -pi/4, 0]) )
        target2 = transform(np.array([0.562-0.1, 0.978-0.809+0.1, 0.200+0.030+0.05*height]), np.array([pi, pi/4, 0]) )
        mid = transform(np.array([0.562-0.1, 0.978-0.809, 0.200+0.080+0.05*height]), np.array([pi, 0, 0]) )
        seedmid = pose_dict["redoffneutral"]
    else:
        seed1 = pose_dict["blueoff1"]
        seed2 = pose_dict["blueoff2"]
        target1 = transform(np.array([0.562-0.1, -0.978+0.809-0.1, 0.200+0.030+0.05*height]), np.array([pi, -pi/4, 0]) )
        target2 = transform(np.array([0.562-0.1, -0.978+0.809-0.1, 0.200+0.030+0.05*height]), np.array([pi, pi/4, 0]) )
        mid = transform(np.array([0.562-0.1, -0.978+0.809, 0.200+0.080+0.05*height]), np.array([pi, 0, 0]) )
        seedmid = pose_dict["blueoffneutral"]

    clearance = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0 ,0 ,0, 0.05], [0, 0, 0, 0]])
    target1a = target1 + clearance
    target2a = target2 + clearance

    q1, success, rollout = ik.inverse(target1, seed1)
    print(success)
    q2, success, rollout = ik.inverse(target2, seed2)
    print(success)

    q1a, success, rollout = ik.inverse(target1a, seed1)
    print(success)
    q2a, success, rollout = ik.inverse(target2a, seed2)
    print(success)

    qmid, success, rollout = ik.inverse(mid, seedmid)
    print(success)

    print("Executing")
    #open
    arm.exec_gripper_cmd(0.1, 0)
    #move to -45
    arm.safe_move_to_position(qmid)
    arm.safe_move_to_position(q1a)
    arm.safe_move_to_position(q1)
    #grab
    arm.exec_gripper_cmd(0)
    #move up .5
    arm.safe_move_to_position(q1a)
    #rotate
    arm.safe_move_to_position(q2a)
    #move down .5
    arm.safe_move_to_position(q2)
    #release
    arm.exec_gripper_cmd(0.1, 0)
    arm.safe_move_to_position(q2a)

    #eventually can create a library of robot poses instead of using iksolver

    return

#############################
##  Transformation Helpers ##
#############################

def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array([
        [ 1,     0,       0,  0 ],
        [ 0, cos(a), -sin(a), 0 ],
        [ 0, sin(a),  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array([
        [ cos(a), 0, -sin(a), 0 ],
        [      0, 1,       0, 0 ],
        [ sin(a), 0,  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array([
        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])

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

    #arm.safe_move_to_position(arm.neutral_position()) # on your mark!


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
    print("Trobot_tag_list")
    print(Trobot_tag_list)

    # get a list of the transformations of the static blocks in the robot frame
    Trobot_block_list = getBlocks(Trobot_tag_list)
    print("Trobot_block_list")
    print(Trobot_block_list)

    Trobot_block_ordered_list = getOrderedBlocks(Trobot_block_list)
    print("Trobot_block_ordereed_list")
    print(Trobot_block_ordered_list)

    map_struct = loadmap("../../maps/emptyMap.txt")
    # create a new list of theight = 0
    #for Trobot_block in Trobot_block_ordered_list:
    Trobot_EE_list, Tblock_EE_list = getListOfPickUpTransformations(Trobot_block_ordered_list[1])
    print("done with PickUpTransformations")
    pickUpBlock(Trobot_EE_list,deepcopy(map_struct),arm)
    print("here")
    #arm.exec_gripper_cmd(0)

    robotPlaceBlock(team, map, arm, 2)
    rotateBlock(team, 2)
        #height = height + 1ransformations of where the end effector needs to
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
    #test configs
