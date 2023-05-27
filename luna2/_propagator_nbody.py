"""
Propagator for full-ephemeris n-body problem
"""

import numpy as np
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
        self.lstar = lstar
        self.tstar = tstar
        self.vstar = lstar/tstar
        return
    
    def solve(self, et0, t_span, x0,
        t_eval=None, rtol=1e-11, atol=1e-11,
    ):
        """Solve IVP with solve_ivp function"""
        # set parameters
        params = [self.mus, self.naif_ids, et0, self.lstar, self.tstar]
        return solve_ivp(
            eom_nbody, t_span, x0, args=(params,),
            t_eval=t_eval, rtol=rtol, atol=atol,
        )