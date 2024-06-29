"""Generic integration method with pygsl"""

import copy
import numpy as np

import pygsl._numobj as numx
from pygsl import odeiv
    
from ._gsl_event import gsl_event_rootfind


def _check_events(prev_checks, et0, t, y, events):
    """Internal function to check for events"""
    curr_checks = np.array([event(et0,t,y) for event in events])
    checks = np.multiply(curr_checks, prev_checks)
    if np.any(checks < 0):
        detected_indices = np.argwhere(checks<0)[0]
        return True, detected_indices[0], curr_checks
    return False, None, curr_checks


def propagate_gsl(
        params,
        eom,
        et0, t_span, x0, 
        t_eval = None,
        eps_abs = 1e-12,
        eps_rel = 1e-14,
        hstart = 1e-6,
        max_iter = 10000000,
        events = None,
        tol_event = 1e-6,
        maxiter_event = 30,
    ):
    """Propagate using pygsl"""
    # initialize integrator
    stepper = odeiv.step_rk8pd
    dimension = len(x0)
    step = stepper(dimension, eom, args = params)
    control = odeiv.control_y_new(step, eps_abs, eps_rel)
    evolve  = odeiv.evolve(step, control, dimension)
    
    # apply solve
    y = copy.deepcopy(x0)
    h = abs(hstart) * np.sign(t_span[1] - t_span[0])

    # event check array
    detection_success = False
    if events is not None:
        # initialize event signs
        prev_checks = np.array([event(et0,0,x0) for event in events])

    if t_eval is None:
        ts = [t_span[0],]       # initialize
        ys = [x0,]              # initialize
        t = t_span[0]
        t1 = t_span[1]
        for i in range(max_iter):
            if abs(t) >= abs(t1):
                break
            t, h, y = evolve.apply(t, t1, h, y)
            if events is not None:
                detected, idx_event, prev_checks = _check_events(
                    prev_checks = prev_checks,
                    et0 = et0,
                    t = t,
                    y = y,
                    events = events,
                )
                if detected:
                    t_event, y_event, detection_success = gsl_event_rootfind(
                        evolve = evolve,
                        event = events[idx_event],
                        et0 = et0,
                        t_bounds = [ts[-1], t],
                        y_bounds = [ys[-1], y],
                        tol = tol_event,
                        maxiter = maxiter_event,
                    )
                    ts.append(t_event)
                    ys.append(y_event)
                    break
            ts.append(t)
            ys.append(y)

    else:
        detected = False        # initialize
        ts = [t_eval[0],]       # initialize
        ys = [x0,]              # initialize
        for i_leg in range(len(t_eval)-1):
            t = t_eval[i_leg]
            t1 = t_eval[i_leg+1]
            for _ in range(max_iter):
                if abs(t) >= abs(t1):
                    break
                t, h, y = evolve.apply(t, t1, h, y)
                if events is not None:
                    detected, idx_event, prev_checks = _check_events(
                        prev_checks = prev_checks,
                        et0 = et0,
                        t = t,
                        y = y,
                        events = events,
                    )
                    if detected:
                        t_event, y_event, detection_success = gsl_event_rootfind(
                            evolve = evolve,
                            event = events[idx_event],
                            et0 = et0,
                            t_bounds = [ts[-1], t],
                            y_bounds = [ys[-1], y],
                            tol = tol_event,
                            maxiter = maxiter_event,
                        )
                        ts.append(t_event)
                        ys.append(y_event)
                        break
            # check if event has been detected
            if detected:
                break
            ts.append(t)
            ys.append(y)

    # post-processing
    ts = np.array(ts)
    ys = np.array(ys).T
    return ts, ys, detection_success