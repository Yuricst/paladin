"""
Propagator with pygsl for MEE
For SPICE inertial frames, see:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Appendix.%20%60%60Built%20in''%20Inertial%20Reference%20Frames
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import pygsl._numobj as numx
from pygsl import odeiv

from ._symbolic_jacobians import get_jaocbian_expr_Nbody
from ._eom_mee import eom_mee
from ._plotter import set_equal_axis
from ._gsl_event import gsl_event_rootfind
from ._ODESolution import PseudoODESolution
from ._propagate_gsl import propagate_gsl


class GSLPropagatorMEE:
    """Spacecraft MEE propagator object"""
    def __init__(
        self,
        naif_frame,
        naif_ids,
        mus,
        lstar = 3000.0,
        use_canonical=False,
        analytical_jacobian = True,
    ):
        """Initialize propagator"""
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
        
        # analytical jacobian 
        self.analytical_jacobian = analytical_jacobian
        if self.analytical_jacobian:
            self.jac_func = get_jaocbian_expr_Nbody(self.mus_use)
        else:
            self.jac_func = None

        # eoms
        self.rhs = eom_mee
        self.rhs_stm = None
        return
    
    def summary(self):
        """Print info about integrator"""
        print(f" ******* MEE GSL propagator summary ******* ")
        print(f" |   NAIF frame      : {self.naif_frame}")
        print(f" |   NAIF IDs        : {self.naif_ids}")
        print(f" |   GMs             : {self.mus}")
        print(f" |   Canonical units : {self.use_canonical}")
        print(f" |   lstar           : {self.lstar}")
        print(f" |   tstar           : {self.tstar}")
        print(f" |   vstar           : {self.vstar}")
        print(f" |   Jacobian        : {self.analytical_jacobian}")
        print(f" ----------------------------------------- ")
        return
    
    def dim2nondim(self, state):
        """Convert dimensional state to caoninical units"""
        assert len(state) == 6, "state should be length 6"
        return np.concatenate(([state[0]/self.lstar,], state[1:]))

    def nondim2dim(self, state):
        """Convert canonical state to dimensional units"""
        assert len(state) == 6, "state should be length 6"
        return np.concatenate(([state[0]*self.lstar,], state[1:]))
    

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
        events = None,
        tol_event = 1e-6,
        maxiter_event = 30,
    ):
        """Solve IVP for state with GSL's rk8pd function
        
        Args:
            et0 (float): initial epoch, in ephemeris seconds
            t_span (list): initial and final integration time
            x0 (np.array): initial state
            t_eval (np.array): times at which to store solution
            method (str): integration method
            eps_abs (float): absolute tolerance
            eps_rel (float): relative tolerance
            hstart (float): initial step size
            max_iter (int): maximum number of iterations
            events (list of callable): event function with syntax `event(et0,t,y)` to track
            
        Returns:
            (bunch object): solution object with properties `t` and `y`
        """
        assert len(x0) == 6, "Initial state should be length 6!"

        # initialize integrator
        params = [self.mus_use, self.naif_ids, et0, self.lstar, self.tstar, self.naif_frame]
    
        # run propagation
        ts, ys, self.detection_success = propagate_gsl(
            params,
            eom_mee,
            et0, t_span, x0, 
            t_eval = t_eval,
            eps_abs = eps_abs,
            eps_rel = eps_rel,
            hstart = hstart,
            max_iter = max_iter,
            events = events,
            tol_event = tol_event,
            maxiter_event = maxiter_event,
        )
        return PseudoODESolution(ts, ys)
    