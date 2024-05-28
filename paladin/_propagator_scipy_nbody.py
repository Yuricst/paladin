"""
Propagator with scipy for full-ephemeris n-body problem
For SPICE inertial frames, see:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Appendix.%20%60%60Built%20in''%20Inertial%20Reference%20Frames
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ._symbolic_jacobians import get_jaocbian_expr_Nbody
from ._eom_scipy_nbody import eom_nbody, eomstm_nbody
from ._plotter import set_equal_axis


class PropagatorNBody:
    """Class for N-body propagator"""
    def __init__(
        self,
        naif_frame,
        naif_ids,
        mus,
        lstar=3000.0,
        use_canonical=False,
        analytical_jacobian = True,
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
        
        # analytical jacobian 
        self.analytical_jacobian = analytical_jacobian
        if self.analytical_jacobian:
            self.jac_func = get_jaocbian_expr_Nbody(self.mus_use)
        else:
            self.jac_func = None
        return
    
    def summary(self):
        """Print info about integrator"""
        print(f" ******* N-body scipy propagator summary ******* ")
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
        t_eval=None,
        method="RK45",
        rtol=1e-11,
        atol=1e-11,
        dense_output=False,
        events=None,
    ):
        """Solve IVP for state with solve_ivp function
        
        Args:
            et0 (float): initial epoch, in ephemeris seconds
            t_span (list): initial and final integration time
            x0 (np.array): initial state
            t_eval (np.array): times at which to store solution
            method (str): integration method
            rtol (float): relative tolerance
            atol (float): absolute tolerance
            dense_output (bool): whether to return dense output
            events (callable, or list of callables, optional): events to track
        
        Returns:
            (bunch object): returned object from `scipy.integrate.solve_ivp`
        """
        # set parameters
        params = [self.mus_use, self.naif_ids, et0, self.lstar, self.tstar, self.naif_frame]
        return solve_ivp(
            eom_nbody, t_span, x0, args=(params,),
            t_eval=t_eval, method=method, rtol=rtol, atol=atol,
            events=events, dense_output=dense_output,
        )
    
    def solve_stm(
        self,
        et0,
        t_span,
        x0,
        stm0 = None,
        t_eval=None,
        method="RK45",
        eps_rel=1e-11,
        eps_abs=1e-11,
        dense_output=False,
        events=None,
    ):
        """Solve IVP for state and STM with solve_ivp function
        
        Args:
            et0 (float): initial epoch, in ephemeris seconds
            t_span (list): initial and final integration time
            x0 (np.array): initial state
            stm0 (np.array): initial STM; if None, identity matrix is used
            t_eval (np.array): times at which to store solution
            method (str): integration method
            eps_rel (float): relative tolerance
            eps_abs (float): absolute tolerance
            dense_output (bool): whether to return dense output
            events (callable, or list of callables, optional): events to track
        
        Returns:
            (bunch object): returned object from `scipy.integrate.solve_ivp`
        """
        assert self.jac_func is not None, "Jacobian function not available!"

        if stm0 is None:
            stm0 = np.eye(6)
        else:
            assert stm0.shape == (6,6), "STM should be 6x6 matrix"

        # set parameters
        params = [self.mus_use, self.naif_ids, et0, self.lstar, self.tstar,
                  self.naif_frame, self.jac_func]
        return solve_ivp(
            eomstm_nbody, t_span, np.concatenate((x0, stm0.flatten())), args=(params,),
            t_eval=t_eval, method=method, rtol=eps_rel, atol=eps_abs, 
            events=events, dense_output=dense_output,
        )
    
    def get_stm_cdm(self, et0, tf, x0, h=1e-6, get_svf=False):
        """Get STM from et0 to tf using x0 as initial state
        
        Args:
            et0 (float): initial epoch, in ephemeris seconds
            tf (real): final time
            x0 (np.array): initial state
            h (float): perturbation magnitude
            get_svf (bool): whether to also return final state
        
        Returns:
            (np.array or tuple): STM, or final state and STM
        """
        stm = np.zeros((6,6))
        for idx in range(6):
            # forward perturbed propagation
            x0_ptrb_fwd = copy.deepcopy(x0)
            x0_ptrb_fwd[idx] += h
            sol_fwd = self.solve(et0, [0,tf], x0_ptrb_fwd)
            # backward perturbed propagation
            x0_ptrb_bck = copy.deepcopy(x0)
            x0_ptrb_bck[idx] -= h
            sol_bck = self.solve(et0, [0,tf], x0_ptrb_bck)
            # store column
            stm[:,idx] = (sol_fwd.y[:,-1] - sol_bck.y[:,-1])/(2*h)
        # return stm only
        if get_svf is False:
            return stm
        else:
            sol = self.solve(et0, [0,tf], x0)
            return sol.y[:,-1], stm, sol
        
    def plot3(self, sol, ax=None, color="deeppink", linewidth=0.7, scale_equal_axis=1.05):
        """Plot 3D trajectory of propagated solution"""
        if ax is None:
            new_plot = True
            fig = plt.figure(figsize = (8,8))
            ax = plt.axes(projection = '3d')
        else:
            new_plot = False
        ax.plot(
            sol.y[0,:]*self.lstar,
            sol.y[1,:]*self.lstar,
            sol.y[2,:]*self.lstar,
            color=color,
            linewidth=linewidth,
        )
        set_equal_axis(
            sol.y[0,:]*self.lstar,
            sol.y[1,:]*self.lstar,
            sol.y[2,:]*self.lstar,
            scale=scale_equal_axis,
        )
        # return figure
        if new_plot:
            return fig, ax
        else:
            return ax
    
