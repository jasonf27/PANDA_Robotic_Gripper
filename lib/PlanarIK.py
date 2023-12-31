import numpy as np
from math import pi


class PlanarIK:
    """
    Solves the planar IK problem for panda robot arm
    """
    offset = 0.0825  # off axis offset
    offset_5 = 0.088
    l0 = 0.333  # length of link from j0 to j1
    l1 = 0.316  # length of link from j1 to j3
    l3 = 0.384  # length of link from j1 to j5
    l5 = 0.107 + 0.11 # length of link from j5 to gripper point

    # angle offsets due to non straight link
    j1_offset = np.arctan(-offset / l1)
    j3_offset = j1_offset + np.arctan(-offset / l3)
    j5_offset = -j3_offset + np.arctan(-offset_5 / l5) + j1_offset

    def panda_ik(self, physical_target):
        """
        Solves planar IK problem given physical target in x,z plane
        Args:
            physical_target: dictionary containing:
                'o': numpy array of target relative to robot base[x,z]
                'theta': scalar of angle about y axis for target in radians

        Returns:
             q = 2x7 numpy array of joints in radians
        """
        abstract_target = self.physical_to_abstract_target(physical_target)
        abstract_joints = self.rrr_abstract_ik(abstract_target)
        physical_joints = self.physical_to_abstract_joints(abstract_joints)
        return self.convert_to_range(physical_joints)

    def physical_to_abstract_target(self, physical_target):
        """
        Converts physical target to abstract target relative to joint 1 in rrr bot configuration in the xy plane.
        0 configuration in abstract space has arm fully extended pointing along the x axis
        Args:
            physical_target: dictionary containing:
                'o': numpy array of target relative to robot base[x,z]
                'theta': scalar of angle about y axis for target

        Returns:
            abstract_target: dictionary containing:
                'o': numpy array of target relative to j1[x,y]
                'theta': scalar of angle about abstract z axis in radians

        """
        abstract_target = physical_target
        abstract_target['o'] = physical_target['o'] - np.array([0, self.l0])  # shift by length l0
        # Change from positive clockwise to positive counter clockwise and shift angle by joint offsets
        abstract_target['theta'] = -physical_target['theta'] - (-self.j1_offset + self.j3_offset + self.j5_offset)
        return abstract_target

    def physical_to_abstract_joints(self, abstract_joints):
        """
        Converts abstract joints to physical joints
        0 configuration in abstract space has arm fully extended pointing along the x axis
        Args:
            abstract_joints: 2x3 np.array in radians

        Returns:
             physical_joints: 2x7 np.array in radians
        """
        # Zero position is abstract arm fully extended along x axis
        zeros = np.tile(np.array([0, pi / 2 + self.j1_offset, 0, self.j3_offset, 0, pi + self.j5_offset, pi / 4]), (2,1))
        # Add abstract joints accounting for different directions
        physical_joints = zeros + np.array([[0, -abstract_joints[0, 0], 0, abstract_joints[0, 1], 0, abstract_joints[0, 2], 0],
                                            [0, -abstract_joints[1, 0], 0, abstract_joints[1, 1], 0, abstract_joints[1, 2], 0]])
        return physical_joints

    def convert_to_range(self, physical_joints):
        """
        Ensures joints are between -pi and pi
        """
        physical_joints = physical_joints % (2 * pi)
        physical_joints[physical_joints > pi] = physical_joints[physical_joints > pi] - 2 * pi
        return physical_joints

    def rrr_abstract_ik(self, target):
        """
        Solves planar (xy) IK problem for RRR robot where target is relative to 1st joint and zere config of arm has arm
        fully extended along the x axis. Should return both possible IK solutions
        Args:
            target: dictionary containing:
                'o': numpy array of target relative to j1[x,y]
                'theta': scalar of angle about abstract z axis in radians (positive counter clockwise)

        Returns:
            q = 2x3 numpy array of the 3 joints of the RRR robot
        """
        a1 = np.sqrt(self.l1**2.0 + self.offset**2.0)
        a2 = np.sqrt(self.l3**2.0 + self.offset**2.0)
        a3 = np.sqrt(self.l5 ** 2.0 + self.offset_5 ** 2.0)

        q1_a = 0
        q2_a = 0
        q3_a = 0
        q1_b = 0
        q2_b = 0
        q3_b = 0
        # **** Student code goes here ****

        o = target["o"]
        theta = target["theta"]
        x_p = o[0] - a3*np.cos(theta)
        y_p = o[1] - a3*np.sin(theta)

        sqr = np.sqrt(x_p**2 + y_p**2)

        gamma = np.arctan2(-1*y_p/sqr, -1*x_p/sqr)
        q1_a = gamma + -1*np.arccos(-1*(x_p**2+y_p**2+a1**2-a2**2)/(2*a1*sqr))
        q2_a = np.arctan2((y_p-a1*np.sin(q1_a))/a2, (x_p-a1*np.cos(q1_a))/a2) - q1_a
        q3_a = theta - (q1_a + q2_a)

        q1_b = gamma + 1*np.arccos(-1*(x_p**2+y_p**2+a1**2-a2**2)/(2*a1*sqr))
        q2_b = np.arctan2((y_p-a1*np.sin(q1_b))/a2, (x_p-a1*np.cos(q1_b))/a2) - q1_b
        q3_b = theta - (q1_b + q2_b)


        return np.array([[q1_a, q2_a, q3_a], [q1_b, q2_b, q3_b]])
