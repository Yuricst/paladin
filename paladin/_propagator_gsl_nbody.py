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

from ._symbolic_jacobians import get_jaocbian_expr_Nbody
from ._eom_scipy_nbody import eom_nbody, eomstm_nbody
from ._plotter import set_equal_axis
from ._gsl_event import gsl_event_rootfind
from ._ODESolution import PseudoODESolution
from ._propagate_gsl import propagate_gsl


class GSLPropagatorNBody:
    """Spacecraft N-body dynamics propagator object wrapping GSL's RK8PD method
    
    The spacecraft's N-body problem is formulated with `naif_ids[0]` as the primary body.
    If `use_canonical = True`, `mus` are scaled such that the `mu` of the primary body is 1.
    
    Args:
        naif_frame (str): SPICE frame name
        naif_ids (list): SPICE NAIF IDs of gravitational bodies to account for
        mus (list): GMs of gravitational bodies to account for, in [km^3/s^2]
        lstar (float): length scale for canonical units, in [km]
        P_srp (float): SRP pressure at 1 AU, in [N/m^2] = [kg/(m.s^2)]
        B_srp (float): SRP reflection coefficient Cr * A/m, in [m^2/kg]
        AU_km (float): Astronomical unit in [km]
        use_canonical (bool): whether to use canonical units
    """
    def __init__(
        self,
        naif_frame,
        naif_ids,
        mus,
        lstar = 3000.0,
        P_srp = 4.56e-6,               # N/m^2
        B_srp = 0.0,                   # Cr * A/m
        AU_km = 149.597870700e6,
        use_canonical = False,
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

        # parameters for SRP
        self.AU = AU_km / lstar
        self.k_srp = (P_srp * B_srp)/1e3 / (self.lstar/self.tstar**2)
        
        # analytical jacobian 
        self.analytical_jacobian = analytical_jacobian
        if self.analytical_jacobian:
            self.jac_func = get_jaocbian_expr_Nbody(self.mus_use)
        else:
            self.jac_func = None

        # eoms
        if B_srp == 0.0:
            self.use_srp = False
            self.rhs = eom_nbody
            self.rhs_stm = eomstm_nbody
        else:
            self.use_srp = True
            raise NotImplementedError
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
        print(f" |   Jacobian        : {self.analytical_jacobian}")
        print(f" ----------------------------------------- ")
        return
    
    def dim2nondim(self, state):
        """Convert dimensional state to caoninical units"""
        assert len(state) == 6, "state should be length 6"
        return np.concatenate((state[0:3]/self.lstar, state[3:6]/self.vstar))

    def nondim2dim(self, state):
        """Convert canonical state to dimensional units"""
        assert len(state) == 6, "state should be length 6"
        return np.concatenate((state[0:3]*self.lstar, state[3:6]*self.vstar))
    
    def eom(self, et0, t, x):
        """Evaluate equations of motion"""
        if self.use_srp is False:
            params = [self.mus_use,
                      self.naif_ids,
                      et0,
                      self.lstar,
                      self.tstar,
                      self.naif_frame]
        else:
            params = [self.mus_use,
                      self.naif_ids,
                      et0,
                      self.lstar,
                      self.tstar, 
                      self.naif_frame,
                      self.AU,
                      self.k_srp]
        return self.rhs(t, x, params)
    
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
        if self.use_srp is False:
            params = [self.mus_use,
                      self.naif_ids,
                      et0,
                      self.lstar,
                      self.tstar,
                      self.naif_frame]
        else:
            params = [self.mus_use,
                      self.naif_ids,
                      et0,
                      self.lstar,
                      self.tstar, 
                      self.naif_frame,
                      self.AU,
                      self.k_srp]
            
        # run propagation
        ts, ys, self.detection_success = propagate_gsl(
            params,
            self.rhs,
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
    

    def solve_stm(
        self,
        et0,
        t_span,
        x0,
        stm0 = None,
        t_eval = None,
        eps_abs = 1e-12,
        eps_rel = 1e-14,
        hstart = 1e-6,
        max_iter = 10000000,
        events = None,
        tol_event = 1e-6,
        maxiter_event = 30,
    ):
        """Solve IVP for state and STM with GSL's rk8pd function
        
        Args:
            et0 (float): initial epoch, in ephemeris seconds
            t_span (list): initial and final integration time
            x0 (np.array): initial state
            stm0 (np.array): initial STM; if None, identity matrix is used
            t_eval (np.array): times at which to store solution
            method (str): integration method
            eps_abs (float): absolute tolerance
            eps_rel (float): relative tolerance
            hstart (float): initial step size
            max_iter (int): maximum number of iterations
            
        Returns:
            (bunch object): solution object with properties `t` and `y`
        """
        assert len(x0) == 6, "Initial state should be length 6!"
        assert self.jac_func is not None, "Jacobian function not available!"

        if stm0 is None:
            stm0 = np.eye(6)
        else:
            assert stm0.shape == (6,6), "STM should be 6x6 matrix"

        # initialize integrator
        if self.use_srp is False:
            params = [self.mus_use,
                      self.naif_ids,
                      et0,
                      self.lstar,
                      self.tstar,
                      self.naif_frame,
                      self.jac_func]
        else:
            params = [self.mus_use,
                      self.naif_ids,
                      et0,
                      self.lstar,
                      self.tstar, 
                      self.naif_frame,
                      self.AU,
                      self.k_srp,
                      self.jac_func]
        x0_aug = np.concatenate((x0, stm0.flatten()))   # augmented state

        # run propagation
        ts, ys, self.detection_success = propagate_gsl(
            params,
            self.rhs_stm,
            et0, t_span, x0_aug, 
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
    
    
    def _check_events(self, prev_checks, et0, t, y, events):
        """Internal function to check for events"""
        curr_checks = np.array([event(et0,t,y) for event in events])
        checks = np.multiply(curr_checks, prev_checks)
        if np.any(checks < 0):
            detected_indices = np.argwhere(checks<0)[0]
            return True, detected_indices[0], curr_checks
        return False, None, curr_checks
    
            
    # def get_xdot(self, et0, t, x):
    #     """Get state-derivative
        
    #     Args:
    #         t (float): time
    #         x (np.array): state-vector
        
    #     Returns:
    #         (np.array): state-derivative
    #     """
    #     if self.use_srp is False:
    #         params = [self.mus_use,
    #                   self.naif_ids,
    #                   et0,
    #                   self.lstar,
    #                   self.tstar,
    #                   self.naif_frame]
    #     else:
    #         params = [self.mus_use,
    #                   self.naif_ids,
    #                   et0,
    #                   self.lstar,
    #                   self.tstar, 
    #                   self.naif_frame,
    #                   self.AU,
    #                   self.k_srp]
    #     return self.rhs(t, x, params)