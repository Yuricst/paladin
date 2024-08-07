"""
Scipy implementation of N-body problem with SRP eom
"""

import numpy as np
import spiceypy as spice
from numba import njit

from .._perturbations import third_body_battin


def eom_nbody_srp(t,states,params):
    """Equations of motion of N-body problem"""
    # unpack states and params
    mus, naif_ids, et0, lstar, tstar, naif_frame, k_srp = params

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
        dstates[3:6] += accel3

        # if body is Sun, also compute SRP
        if (naif_ids[idx+1] == "10") or (naif_ids[idx+1] == "SUN"):
            rvec_sun = rvec3[0:3]/lstar
            r_relative = states[0:3] - rvec_sun
            dstates[3:6] += k_srp*r_relative/np.linalg.norm(r_relative)**3
    return dstates


def eomstm_nbody_srp(t,states,params):
    """Equations of motion of N-body problem with STM"""
    # unpack states and params
    mus, naif_ids, et0, lstar, tstar, naif_frame, k_srp, jac_func = params

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
        dstates[3:6] += accel3

        # if body is Sun, also compute SRP
        if (naif_ids[idx+1] == "10") or (naif_ids[idx+1] == "SUN"):
            rvec_sun = rvec3[0:3]/lstar
            r_relative = states[0:3] - rvec_sun
            dstates[3:6] += k_srp*r_relative/np.linalg.norm(r_relative)**3
            
        # store for STM derivative copmutation later
        pos_3bd_list[idx,:] = rvec3[0:3]/lstar
    
    # Compute STM derivative
    dstates[6:] = (jac_func(states[0:6], mus, pos_3bd_list, k_srp, rvec_sun) @ (states[6:].reshape(6,6))).reshape(36,)
    return dstates
