"""
Scipy implementation of N-body problem eom
"""

import numpy as np
import spiceypy as spice
from numba import njit


@njit
def third_body_battin(r, s, mu_third):
    """Third body perturbation acceleration via Battin's formulation"""
    d = r - s
    dnorm = np.sqrt(np.dot(d,d))
    q = np.dot(r, r - 2*s)/np.dot(s,s)
    F = q*((3 + 3*q + q**2)/(1 + np.sqrt(1+q)**3))
    return -mu_third/dnorm**3 * (r + F*s)


def eom_nbody(t,states,params):
    """Equations of motion of N-body problem"""
    # unpack states and params
    mus, naif_ids, et0, lstar, tstar, naif_frame = params

    # initialize derivatives
    dstates = np.zeros(6,)
    dstates[0:3] = states[3:6]
    dstates[3:6] = -mus[0]/np.linalg.norm(states[0:3])**3 * states[0:3]

    # Iterate through extra bodies
    for idx in range(len(mus)-1):
        rvec3, _ = spice.spkpos(
            naif_ids[idx+1], et0 + t*tstar, 
            naif_frame, "NONE", naif_ids[0]
        )
        accel3 = third_body_battin(
            states[0:3], rvec3/lstar, mus[idx+1]
        )
        # Add to accelerations
        dstates[3:6] += accel3
    return dstates


def eomstm_nbody(t,states,params):
    """Equations of motion of N-body problem with STM"""
    # unpack states and params
    mus, naif_ids, et0, lstar, tstar, naif_frame, jac_func = params

    # initialize derivatives
    dstates = np.zeros((42,))
    dstates[0:3] = states[3:6]
    dstates[3:6] = -mus[0]/np.linalg.norm(states[0:3])**3 * states[0:3]

    # Iterate through extra bodies
    pos_3bd_list = np.zeros((len(mus)-1,3))
    for idx in range(len(mus)-1):
        rvec3, _ = spice.spkpos(
            naif_ids[idx+1], et0 + t*tstar, 
            naif_frame, "NONE", naif_ids[0]
        )
        accel3 = third_body_battin(
            states[0:3], rvec3/lstar, mus[idx+1]
        )
        # Add to accelerations
        dstates[3:6] += accel3
        # store for STM derivative copmutation later
        pos_3bd_list[idx,:] = rvec3[0:3]/lstar
    
    # Compute STM derivative
    dstates[6:] = (jac_func(states[0:6], mus, pos_3bd_list) @ (states[6:].reshape(6,6))).reshape(36,)
    return dstates
