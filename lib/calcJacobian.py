import numpy as np
from lib.calculateFK import FK

def calcJacobian(q):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q: 0 x 7 configuration vector (of joint angles) [q0,q1,q2,q3,q4,q5,q6]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    #Compute Jv1
    joint_positions, T0e, T0x = FK.forward(q, q, 2)
    on = np.array([T0e[0,3],T0e[1,3],T0e[2,3]])
    oi = np.array([joint_positions[0,0],joint_positions[0,1],joint_positions[0,2]])
    o = (on - oi)
    z = np.array([0,0,1])
    J[0:3,0] = np.cross(z,o)
    J[3:6,0] = np.array([0, 0, 1])


    for x in range(1,6):
        #CALCULATE FK OF GIVEN SYSTEM
        joint_positions, T0e, T0x = FK.forward(q, q, x)
        #VECTOR OF DISTANCE JOINT X TO ORIGIN FROM JOINT POSITIONS MATRIX
        oi = np.array([joint_positions[x,0],joint_positions[x,1],joint_positions[x,2]])
        #CALCULATE DIFFERENCE BETWEEN TWO VECTORS
        o = (on - oi)
        #CALCULATE z-axis of frame i-1 in terms of origin frame
        z = np.array([T0x[0,2],T0x[1,2],T0x[2,2]])
        #Add cross product into Jacobian matrix
        J[0:3,x] = np.cross(z,o)
        #Add in z-axis of coordinate frame into angular velocity jacobian
        if x == 3 or x == 5:
            J[0:3,x] = -np.cross(z,o)
            J[3:6,x] = -z
        else:
            J[0:3,x] = np.cross(z,o)
            J[3:6,x] = z

    joint_positions, T0e, T0x = FK.forward(q, q, 6)
    z = np.array([T0x[0,2],T0x[1,2],T0x[2,2]])
    J[3:6,6] = z


    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
