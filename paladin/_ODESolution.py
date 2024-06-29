"""
ODE solution pseudo-object
"""


class PseudoODESolution:
    """Class for storing time and states in a format similar to ODE solution from solve_ivp.
    This is purely for interoperability with EKF.
    """
    def __init__(self, ts, y, t_events=None, y_events=None):
        assert len(ts) == y.shape[1],\
            f"y should be nx-by-N (y.shape = {y.shape}), where N is time-steps (len(ts) = {len(ts)})"
        self.t = ts 
        self.y = y
        self.t_events = t_events
        self.y_events = y_events
        return 