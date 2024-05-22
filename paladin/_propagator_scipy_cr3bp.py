"""
Propagator with scipy for CR3BP
"""

import numpy as np
from scipy.integrate import solve_ivp
from ._eom_scipy_cr3bp import eom_cr3bp, eom_cr3bp_stm

class PropagatorCR3BP:
    """Class for N-body propagator"""
    def __init__(
        self,
        mu,
        get_stm=False
    ):
        """Initialize propagator
        
        Args:
            mu (float): mass parameter in the CR3BP
            get_stm (bool): whether to also propagate STM
        """
        self.mu = mu
        self.get_stm = get_stm
        return
    
    def solve(self, t_span, x0,
        t_eval=None, rtol=1e-11, atol=1e-11, dense_output=False
    ):
        """Solve IVP with solve_ivp function"""
        if self.get_stm:
            eom = eom_cr3bp_stm
        else:
            eom = eom_cr3bp
        return solve_ivp(
            eom, t_span, x0, args=(self.mu,),
            t_eval=t_eval, rtol=rtol, atol=atol, dense_output=dense_output,
        )