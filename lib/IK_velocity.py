import numpy as np
from lib.calcJacobian import calcJacobian

def IK_velocity(q_in, v_in, omega_in):
    """
    :param q: 0 x 7 vector corresponding to the robot's current configuration.
    :param v: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 0 x 7 vector corresponding to the joint velocities. If v and omega
         are infeasible, then dq should minimize the least squares error. If v
         and omega have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros(7)

    J = calcJacobian(q_in)

    #Find and remove NaN's, form shi
    #Remove rows from J
    shi, J = get_shi(J, v_in, omega_in)

    #rank test
    """
    rank_test_mat = np.concatenate(J, shi, axis=1)
    if (np.linalg.matrix_rank(J) == np.linalg.matrix_rank(rank_test_mat)):
        dq = analytical(J, shi)
    else:
        dq = np.linalg.lstsq(J, shi)
    """
    dq = np.linalg.lstsq(J, shi)[0]

    return dq

## Subfunctions
def get_shi(J, v_in, omega_in):

    shi2 = np.array([])
    J2 = np.array([])
    shi = np.concatenate((v_in, omega_in))
    for idx, val in enumerate(np.isnan(shi)):
        if (val == False):
            shi2 = np.append(shi2, shi[idx])
            J2 = np.append(J2, J[idx,:], axis=0)
    J2 = np.reshape(J2, (int(J2.size/7), 7))

    return shi2, J2

def analytical(J, shi):
    return 0
