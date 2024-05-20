"""
Propagator with pygsl for full-ephemeris n-body problem
For SPICE inertial frames, see:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Appendix.%20%60%60Built%20in''%20Inertial%20Reference%20Frames
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import pygsl._numobj as numx
from pygsl import odeiv


from ._eom_scipy_nbody import eom_nbody
from ._plotter import set_equal_axis


class PseudoODESolution:
    """Class for storing time and states in a format similar to ODE solution from solve_ivp.
    This is purely for interoperability with EKF.
    """
    def __init__(self, ts, y, t_events=None, y_events=None):
        assert len(ts) == y.shape[1], "y should be nx-by-N, where N is time-steps"
        self.t = ts 
        self.y = y
        self.t_events = t_events
        self.y_events = y_events
        return 


class GSLPropagatorNBody:
    def __init__(
        self,
        naif_frame,
        naif_ids,
        mus,
        lstar = 3000.0,
        use_canonical=False,
    ):
        """Initialize propagator.
        The spacecraft's N-body problem is formulated with `naif_ids[0]` as the primary body.
        If `use_canonical = True`, `mus` are scaled such that the `mu` of the primary body is 1.
        
        Args:
            naif_frame (str): SPICE frame name
            naif_ids (list): SPICE NAIF IDs of gravitational bodies to account for
            mus (list): GMs of gravitational bodies to account for, in km^3/s^2
            lstar (float): length scale for canonical units, in km
            use_canonical (bool): whether to use canonical units
        """
        self.naif_frame = naif_frame
        self.naif_ids = naif_ids
        self.mus = mus
        self.use_canonical = use_canonical
        if use_canonical:
            self.mus_use = np.array(self.mus) / mus[0]
            self.lstar = lstar
            self.vstar = np.sqrt(mus[0]/self.lstar)
            self.tstar = lstar/self.vstar
        else:
            self.mus_use = self.mus
            self.lstar, self.tstar, self.vstar = 1.0, 1.0, 1.0
        return
    
    def summary(self):
        """Print info about integrator"""
        print(f" ******* N-body GSL propagator summary ******* ")
        print(f" |   NAIF frame      : {self.naif_frame}")
        print(f" |   NAIF IDs        : {self.naif_ids}")
        print(f" |   GMs             : {self.mus}")
        print(f" |   Canonical units : {self.use_canonical}")
        print(f" |   lstar           : {self.lstar}")
        print(f" |   tstar           : {self.tstar}")
        print(f" |   vstar           : {self.vstar}")
        print(f" ----------------------------------------- ")
        return
    
    def dim2nondim(self, state):
        assert len(state) == 6, "state should be length 6"
        return np.concatenate((state[0:3]/self.lstar, state[3:6]/self.vstar))

    def eom(self, et0, t, x):
        """Evaluate equations of motion"""
        params = [self.mus_use, self.naif_ids, et0, self.lstar, self.tstar, self.naif_frame]
        return eom_nbody(t, x, params)
    
    def solve(
        self,
        et0,
        t_span,
        x0,
        t_eval = None,
        eps_abs = 1e-12,
        eps_rel = 1e-14,
        hstart = 1e-6,
        max_iter = 10000000,
    ):
        assert len(x0) == 6, "Initial state should be length 6!"

        # initialize integrator
        params = [self.mus_use, self.naif_ids, et0, self.lstar, self.tstar, self.naif_frame]
        stepper = odeiv.step_rk8pd
        dimension = len(x0)
        step = stepper(dimension, eom_nbody, args = params)
        control = odeiv.control_y_new(step, eps_abs, eps_rel)
        evolve  = odeiv.evolve(step, control, dimension)

        # initialize storage
        ts = []
        ys = []

        # apply solve
        t = t_span[0]
        t1 = t_span[1]
        y = copy.deepcopy(x0)
        h = hstart
        for i in range(max_iter):
            if t >= t1:
                break
            t, h, y = evolve.apply(t, t1, h, y)
            ts.append(t)
            ys.append(y)
        ys = np.array(ys).T
        return PseudoODESolution(ts, ys)