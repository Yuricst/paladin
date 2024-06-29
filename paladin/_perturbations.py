"""
Perturbation functions
"""

import numpy as np
from numba import njit


@njit
def third_body_battin(r, s, mu_third):
    """Third body perturbation acceleration via Battin's formulation"""
    d = r - s
    dnorm = np.sqrt(np.dot(d,d))
    q = np.dot(r, r - 2*s)/np.dot(s,s)
    F = q*((3 + 3*q + q**2)/(1 + np.sqrt(1+q)**3))
    return -mu_third/dnorm**3 * (r + F*s)

