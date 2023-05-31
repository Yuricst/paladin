"""
Propagator for full-ephemeris n-body problem
For SPICE inertial frames, see:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Appendix.%20%60%60Built%20in''%20Inertial%20Reference%20Frames
"""

import numpy as np
import copy
from scipy.integrate import solve_ivp
from ._eom_scipy_nbody import eom_nbody

class PropagatorNBody:
    """Class for N-body propagator"""
    def __init__(
        self,
        naif_frame,
        naif_ids,
        mus,
        lstar,
        tstar,
        use_canonical=False,
    ):
        """Initialize propagator"""
        self.naif_frame = naif_frame
        self.naif_ids = naif_ids
        self.mus = mus
        self.use_canonical = use_canonical
        if use_canonical:
            self.lstar = lstar
            self.tstar = tstar
            self.vstar = lstar/tstar
        else:
            self.lstar, self.tstar, self.vstar = 1.0, 1.0, 1.0
        return
    
    def solve(self, et0, t_span, x0,
        t_eval=None, rtol=1e-11, atol=1e-11, dense_output=False
    ):
        """Solve IVP with solve_ivp function"""
        # set parameters
        params = [self.mus, self.naif_ids, et0, self.lstar, self.tstar]
        return solve_ivp(
            eom_nbody, t_span, x0, args=(params,),
            t_eval=t_eval, rtol=rtol, atol=atol, dense_output=dense_output,
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
            x0_ptrb_bck[idx] += h
            sol_bck = self.solve(et0, [0,tf], x0_ptrb_bck)
            # store column
            stm[:,idx] = (sol_fwd.y[:,-1] - sol_bck.y[:,-1])/(2*h)
        # return stm only
        if get_svf is False:
            return stm
        else:
            sol = self.solve(et0, [0,tf], x0)
            return sol.y[:,-1], stm
    
