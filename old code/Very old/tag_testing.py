import numpy as np
from math import pi

import sys

sys.path.append("/Users/appollo_liu/Documents/workspace/school/meam-520/meam520_labs/")

# The library you implemented over the course of this semester!
from partner_lib.calculateFK import FK
from partner_lib.calcJacobian import FK
from partner_lib.solveIK import IK
from partner_lib.rrt import rrt
from partner_lib.loadmap import loadmap

from tagSampleData import test_tags
from transformation_helpers import transform
static_tag_names = {"tag1", "tag2", "tag3", "tag4", "tag5", "tag6"}

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
            Ttag0_tagi = np.matmul(Ttag0_camera, pose)
            Trobot_tagi = np.matmul(Trobot_tag0, Ttag0_tagi)

            Trobot_tag_list.append((name, Trobot_tagi))

            # print(name,'\n',Trobot_tagi)

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

if __name__ == "__main__":
    # must be red or blue
    team = "red" 
    # team = "blue" 

    Trobot_tag_list = getTags(team, test_tags)
    print("Trobot_tag_list = ", Trobot_tag_list)

    Trobot_block_list = getBlocks(Trobot_tag_list)
    print("Trobot_block_list = ", Trobot_block_list)

    # Trobot_tag1 = transform([0.025, 0, 0], [pi/2, 0, pi/2])
    # Trobot_tag2 = transform([0, 0.025, 0], [pi/2, pi, 0])
    # Trobot_tag3 = transform([-0.025, 0, 0], [pi/2, -pi/2, 0])

    # print("Trobot_tag1 = ", Trobot_tag1)