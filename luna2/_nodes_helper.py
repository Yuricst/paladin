"""
Helper function for nodes handling
"""

import numpy as np


def get_node_bounds_relative(node, fractions):
    """Get node bounds based on relative fraction of state"""
    assert len(node) == len(fractions), "node and fractions must have same length"
    diffs = np.array(fractions) * abs(np.array(node))
    lb = np.array(node) - diffs
    ub = np.array(node) + diffs
    return [lb, ub]