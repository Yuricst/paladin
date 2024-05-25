"""
Object to wrap propagator for filter and SK codes

Compatible with:
- https://github.gatech.edu/SSOG/python-filter
    - Requires `solve(tspan, self._x, stm=True, t_eval = t_eval)` method
- https://github.gatech.edu/SSOG/python-stationkeeping
    - Requires `solve(tspan, x_iter, events=(control_check_event,))` method
"""


class PropagatorWrapper:
    """Wrap to propagator for generic compatibility with other code bases.
    Appended method is `solve(t_span, x0)`, which makes call to the ingerited propagator.

    Args:
        propagator (object): propagator object, e.g. paladin.GSLPropagatorNBody
        et0 (float): initial epoch, in ephemeris seconds
        eps_abs (float): absolute tolerance
        eps_rel (float): relative tolerance
    """
    def __init__(self, propagator, et0, eps_abs=1e-12, eps_rel=1e-14):
        self.propagator = propagator
        self.et0 = et0
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        self.rhs = propagator.rhs
        self.rhs_stm = propagator.rhs_stm
        return
    
    def solve(self, tspan, x0, stm = False, t_eval = None, events = None):
        if stm == False:
            sol = self.propagator.solve(
                et0 = self.et0,
                t_span = tspan,
                x0 = x0,
                t_eval = t_eval,
                eps_abs = self.eps_abs,
                eps_rel= self.eps_rel,
                events = events,
            )
        else:
            sol = self.propagator.solve_stm(
                et0 = self.et0,
                t_span = tspan,
                x0 = x0,
                t_eval = t_eval,
                eps_abs = self.eps_abs,
                eps_rel= self.eps_rel,
                events = events,
            )
        return sol
    
    def get_xdot(self, t, x):
        """Get state-derivative
        
        Args:
            t (float): time
            x (np.array): state-vector
        
        Returns:
            (np.array): state-derivative
        """
        return self.propagator.get_xdot(self.et0, t, x)