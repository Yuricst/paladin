"""
Perturbation functions
"""

import numpy as np
from numba import njit


@njit
def third_body_battin(r, s, mu_third):
    """Third body perturbation acceleration via Battin's formulation
    
    Args:
        r (np.array): position vector of spacecraft w.r.t. primary body
        s (np.array): position vector of third body w.r.t. primary body
        mu_third (float): GM of third body

    Returns:
        (np.array): third-body perturbation acceleration
    """
    d = r - s
    dnorm = np.sqrt(np.dot(d,d))
    q = np.dot(r, r - 2*s)/np.dot(s,s)
    F = q*((3 + 3*q + q**2)/(1 + np.sqrt(1+q)**3))
    return -mu_third/dnorm**3 * (r + F*s)


def solar_radiation_pressure(r_sun2sc, AU, k_srp):
    """Solar radiation pressure acceleration
    
    Args:
        r_sun2sc (np.array): position vector of spacecraft w.r.t. Sun, in LU
        AU (float): Astronomical unit, in LU
        k_srp (float): acceleration magnitude, LU/TU^2

    Returns:
        (np.array): SRP acceleration
    """
    return k_srp * AU**2  * r_sun2sc / np.linalg.norm(r_sun2sc)**3