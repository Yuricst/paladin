"""
Propagator for full-ephemeris n-body problem
"""

import numpy as np


class PropagatorNBody:
    """Class for N-body propagator"""
    def __init__(
        self,
        naif_frame,
        lstar,
        tstar,
        use_canonical=False,
    ):
        """Initialize propagator"""
        self.lstar = lstar
        self.tstar = tstar
        self.vstar = lstar/tstar
        return
    
    def solve(self, et0, x0, tof):
        return