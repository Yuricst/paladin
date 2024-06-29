"""
Scipy implementation of eom in MEE
"""

import numpy as np
import spiceypy as spice
from numba import njit

from ._perturbations import third_body_battin


@njit
def mee2rv(mee, mu):
    """Convert MEE to r and v vectors
    
    Args:
        mee (np.array): MEE vector
        mu (float): GM of primary body

    Returns:
        (tuple): position and velocity vectors
    """
    # unpack state
    p,f,g,h,k,L = mee

    # trig eval
    cL = np.cos(L)
    sL = np.sin(L)
    w  = 1. + f * cL + g * sL
    s2 = 1. + h*h + k*k
    r  = p / w

    # convert MEE to pos_vec and vel_vec
    basis1 = np.array([1 - k^2 + h^2, 2 * h * k, -2 * k]) / s2
    basis2 = np.array([2 * h * k, 1 + k^2 - h^2, 2 * h]) / s2
    X = r * cL
    Y = r * sL
    A = np.sqrt(mu / p)
    Xdot = -A * (g + sL)
    Ydot =  A * (f + cL)
    pos_vec = X * basis1 + Y * basis2
    vel_vec = Xdot * basis1 + Ydot * basis2
    return pos_vec, vel_vec


def eom_mee(t,states,params):
    """Equations of motion in terms of MEE"""
    # unpack parameters
    mus, naif_ids, et0, lstar, tstar, naif_frame = params
    mu = mus[0]

    # unpack state
    p,f,g,h,k,L = states

    # trig eval
    cL = np.cos(L)
    sL = np.sin(L)
    w  = 1. + f * cL + g * sL
    wL = g*cL - f*sL
    s2 = 1. + h*h + k*k
    r  = p / w
    e  = np.sqrt(f**2 + g**2)
    a  = p/(1-e**2)
    sqrtpmu = np.sqrt(p/mu)

    # Construct the matrix B in terms of MEE
    B  = sqrtpmu * np.array([
        [ 0.0,   2 * p / w,       0.0               ],
        [ sL,    ((1+w)*cL+f)/w, -g/w * (h*sL-k*cL) ],
        [-cL,    ((1+w)*sL+g)/w,  f/w * (h*sL-k*cL) ],
        [ 0.0,   0.0,             s2/2./w*cL        ],
        [ 0.0,   0.0,             s2/2./w*sL        ],
        [ 0.0,   0.0,             (h*sL-k*cL) / w   ],
    ])
    D = np.array([0,0,0,0,0,np.sqrt(mu*p)*(w/p)**2])

    # convert MEE to pos_vec and vel_vec
    basis1 = np.array([1 - k^2 + h^2, 2 * h * k, -2 * k]) / s2
    basis2 = np.array([2 * h * k, 1 + k^2 - h^2, 2 * h]) / s2
    X = r * cL
    Y = r * sL
    A = np.sqrt(mu / p)
    Xdot = -A * (g + sL)
    Ydot =  A * (f + cL)
    pos_vec = X * basis1 + Y * basis2
    vel_vec = Xdot * basis1 + Ydot * basis2

    # transformation from inertial to RTN
    iR = pos_vec / np.linalg.norm(pos_vec)
    iN = np.cross(pos_vec, vel_vec)
    iN = iN / np.linalg.norm(iN)
    iT = np.cross(iN,iR)
    T_Inr2RTN = np.array([iR, iT, iN])
                        
    # initialize perturbations
    ptrb = np.zeros(3,)

    # third-body perturbations
    for idx in range(len(mus)-1):
        rvec3, _ = spice.spkpos(
            naif_ids[idx+1], et0 + t*tstar, 
            naif_frame, "NONE", naif_ids[0]
        )
        accel3 = third_body_battin(
            states[0:3], rvec3/lstar, mus[idx+1]
        )
        # Add to accelerations
        ptrb += T_Inr2RTN @ accel3

    # state derivatives
    dstates = np.dot(B, ptrb) + D
    return dstates