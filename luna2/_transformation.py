"""
Functions for state transformation
SPICE based routines are used whenever possible
"""

import numpy as np 
import spiceypy as spice


def shift_origin_x(states, shift_x):
    """Shift origin of states by shift_x"""
    # check shape of states
    nrow, ncol = states.shape
    if nrow == 6:
        shifted_xs = states[0,:] + shift_x
        shifted_states = np.concatenate((shifted_xs.reshape(1,-1), states[1:,:]))
    elif ncol == 6:
        shifted_xs = states[:,0] + shift_x
        shifted_states = np.concatenate((shifted_xs.reshape(-1,1), states[:,1:]))
    return shifted_states


def shift_barycenter_to_m2(states, mu):
    """Shift origin of states to from CR3BP barycenter to M2
    Applies shift_origin_x() to states with shift_x = -(1-mu)
    
    Args:
        states (array): states centered at CR3BP barycenter, dimension (6, N)
        mu (float): CR3BP mu

    Returns:
        (np.array): states centered at M2, dimension (6, N)
    """
    return shift_origin_x(states, -(1-mu))


def apply_frame_transformation(epochs, states, frame0, frame1):
    """Apply state transformation matrix at epoch to states
    Transformation matrix is constructed with spice.sxform()
    
    Args:
        epochs (array): epochs in ephemeris seconds, length N
        states (array): states in frame0, dimension (6, N)
        frame0 (str): NAIF name of current frame
        frame1 (str): NAIF name of target frame

    Returns:
        (np.array): states in `frame1`, dimension (6, N)
    """
    nrow, ncol = states.shape
    assert nrow == 6, "States must have 6 rows (x,y,z,vx,vy,vz)"
    assert len(epochs) == ncol, "Epochs and states must have same number of columns"
    # get transformation matrices
    Ts = spice.sxform(frame0, frame1, epochs)  # dimension (N, 6, 6)
    # apply transformation
    return np.einsum('ijk, ki-> ji', Ts, states)


def canonical_to_dimensional(states, lstar, vstar):
    """Convert states in canonical units to states in dimensional units"""
    return np.concatenate((states[0:3,:]*lstar, states[3:6]*vstar))


def dimensional_to_canonical(states, lstar, vstar):
    """Convert states in dimensional units to states in canonical units"""
    return np.concatenate((states[0:3,:]/lstar, states[3:6]/vstar))