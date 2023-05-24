"""
Scipy implementation of N-body problem
"""

import numpy as np
import spiceypy as spice


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
    x,y,z,vx,vy,vz = states
    mus, naif_ids, et0, lstar, tstar = params

    # initialize derivatives
    dstates = [
        vx, vy, vz,
        -mus[0]*x/((x**2+y**2+z**2)**(3/2)),
        -mus[0]*y/((x**2+y**2+z**2)**(3/2)),
        -mus[0]*z/((x**2+y**2+z**2)**(3/2)),
    ]

    # Iterate through extra bodies
    for idx in range(len(mus)-1):
        rvec3, _ = spice.spkpos(
            naif_ids[idx+1], et0 + t/tstar, 
            "J2000", "NONE", naif_ids[0]
        )
        accel3 = third_body_battin(
            np.array([x,y,z]), rvec3/lstar, mus[idx+1]
        )
        # Add to accelerations
        dstates[3:6] += accel3
    return dstates