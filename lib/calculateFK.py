import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        pass


    def forward(self, q, index=6):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]
        index - index of joint to caculate homogenous transformation matrix for

        OUTPUTS:
        jointPositions - 7 x 3 matrix, where each row corresponds to a rotational joint of the robot
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your code starts here
        # manually hardcode each homogenous transformation matrix between consecutive frames based on joint angle inputs
        h01 = np.array([[np.cos(q[0]),-np.sin(q[0]),0,0],
                        [np.sin(q[0]),np.cos(q[0]),0,0],
                        [0,0,1,0.141],
                        [0,0,0,1]])

        h12 = np.array([[np.cos(q[1]),-np.sin(q[1]),0,0],
                        [0,0,1,0],
                        [-np.sin(q[1]),-np.cos(q[1]),0,0.192],
                        [0,0,0,1]])

        h23 = np.array([[np.cos(q[2]),-np.sin(q[2]),0,0],
                        [0,0,-1,-0.195],
                        [np.sin(q[2]),np.cos(q[2]),0,0],
                        [0,0,0,1]])

        h34 = np.array([[-np.sin(q[3]),np.cos(q[3]),0,0.082],
                        [0,0,1,0],
                        [np.cos(q[3]),np.sin(q[3]),0,0.121],
                        [0,0,0,1]])

        h45 = np.array([[0,0,1,0.125],
                        [np.cos(q[4]),-np.sin(q[4]),0,-0.083],
                        [np.sin(q[4]),np.cos(q[4]),0,0],
                        [0,0,0,1]])

        h56 = np.array([[np.sin(q[5]),-np.cos(q[5]),0,0],
                        [0,0,1,0],
                        [-np.cos(q[5]),-np.sin(q[5]),0,0.259],
                        [0,0,0,1]])

        h67 = np.array([[0,0,1,0.051],
                        [-np.cos(q[6]-(pi/4)),np.sin(q[6]-(pi/4)),0,-0.088],
                        [-np.sin(q[6]-(pi/4)),-np.cos(q[6]-(pi/4)),0,0],
                        [0,0,0,1]])

        h7e = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0.159],
                        [0,0,0,1]])

        # put all transformation matrices into one array
        h = [h01, h12, h23, h34, h45, h56, h67, h7e]

        jointPositions = np.zeros([7,3])

        # calculate position of each joint relative to the origin
        # for joint i, multiply all previous homogenous transformation matrices until have h0i, then multiply by p vector [0,0,0,1]
        #define first joint position, hcomp is composition of all homogeneous transformation matrices
        hcomp = h01
        po = np.array([[0,0,0,1]]).T
        pint = np.dot(hcomp,po)
        pf = pint[0:3]
        # add results to final joint positions matrix
        jointPositions[0,0] = pf[0]
        jointPositions[0,1] = pf[1]
        jointPositions[0,2] = pf[2]
        #loop to update hcomp with next rotation matrix, find location of corresponding joint
        for x in range(1,7):
            hcomp = np.dot(hcomp,h[x])
            if x == index:
            	T0i = hcomp
            pint = np.dot(hcomp,po)
            pf = pint[0:3]
            #filter out any very small values that arise by multiplying by -0 (python errors)
            for y in range(3):
                if pf[y]>0.0005 or pf[y]<-0.005:
                    jointPositions[x,y] = pf[y]
                else:
                    jointPositions[x,y] = 0

        #multiply by transformation matrix of end effector to get full homogeneous transformation matrix
        T0e = np.dot(hcomp,h7e)
        #filter out any very small values that arise by multiplying by -0 (python errors)
        #for x in range(4):
         #   for y in range(4):
         #       if T0e[x,y]<0.005 and T0e[x,y]>-0.005:
          #          T0e[x,y] = 0
          #      if T0i[x,y]<0.005 and T0i[x,y]>-0.005:
           #         T0i[x,y] = 0



        # Your code ends here

        return jointPositions, T0e, T0i

    # feel free to define additional helper methods to modularize your solution

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e, T0i = fk.forward(q, 2)

    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
    print("Joint i Pose:\n",T0i)
