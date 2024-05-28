"""Event function handling with GSL integrator"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import pygsl._numobj as numx
from pygsl import odeiv


def gsl_event_rootfind(evolve, event, et0, t_bounds, y_bounds, tol, maxiter):
    """Event function via root-finding with GSL integrator using regula-falsi. 
    
    Args:
        evolve (pygsl.odeiv.evolve): GSL ODE integrator
        event (function): event function
        et0 (float): initial time
        t_bounds (tuple): lower and upper bounds for time
        y_bounds (tuple): lower and upper bounds for state
        tol (float): tolerance for event function
        maxiter (int): maximum number of iterations
    
    Returns:
        (tuple): event time (float), event state (list), detection success flag (bool)
    """
    # step back solution
    t0 = t_bounds[0]
    y0 = y_bounds[0]
    sign0 = event(et0,t0,y0)

    t1 = t_bounds[1]
    y1 = y_bounds[1]
    sign1 = event(et0,t1,y1)

    detection_success = False   # initialize

    # initial guess for step size
    h_try = (t_bounds[1] - t_bounds[0])/2

    for it in range(maxiter):
        t_try, _, y_try = evolve.apply(t0, t0+h_try, h_try, y0)
        sign_try = event(et0,t_try,y_try)
        #print(f"Iter {it} ... {sign_try:1.3e}")
        #print(f"   t0 = {t0:1.3e}, t_try = {t_try:1.3e}, t1 = {t1:1.3e}")
        #print(f"   sign0 = {sign0:1.3e}, sign_try = {sign_try:1.3e}, sign1 = {sign1:1.3e}")

        if np.linalg.norm(sign_try) <= tol:
            #print(f"Event root ({np.linalg.norm(sign_try):1.3e}) found at iter {it}")
            detection_success = True
            break

        # we under-stepped, so we swap the lower bound
        if sign0 * sign_try > 0:
            # work out the intersection via linear approximation
            # between points:
            # (t_try, sign_try) and (t1, sign1)
            slope = (sign1 - sign_try)/(t1 - t_try)
            b = sign_try - slope * t_try
            h_try = -b/slope - t_try

            t0 = t_try
            y0 = y_try
            sign0 = sign_try
            #print(f"   Replacing t0")

        # we over-stepped, so we swap the upper bound
        else:
            # work out the intersection via linear approximation
            # between points:
            # (t0, sign0) and (t_try, sign_try)
            slope = (sign0 - sign_try)/(t0 - t_try)
            b = sign_try - slope * t_try
            h_try = -b/slope - t0

            t1 = t_try
            y1 = y_try
            sign1 = sign_try
            #print(f"   Replacing t1")
    return t_try, y_try, detection_success