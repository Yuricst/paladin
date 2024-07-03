"""
Test for CR3BP propagator
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import paladin


def test_propagator_cr3bp(make_plot = False):
    # list of mus
    mu = 1.215058560962404E-2
    # ID723 in southern halo
    x0 = np.array([
        1.0134037728554581E+0, 0, -1.7536227281091840E-1,
        0, -8.3688427472776439E-2, 0
    ])
    period = 1.3960732332950263E+0

    # create propagator
    prop = paladin.PropagatorCR3BP(mu)
    tof = 2*period
    t_eval = np.linspace(0,tof,1000)
    res = prop.solve([0,tof], x0, t_eval=t_eval)

    if make_plot:
        # shift to Moon-centered coordinates
        states_shifted = paladin.shift_barycenter_to_m2(res.y, mu)
        # plot figure
        fig = plt.figure(figsize = (6, 6))
        ax = plt.axes(projection = '3d')
        ax.plot(states_shifted[0,:], states_shifted[1,:], states_shifted[2,:])
    assert all(np.abs(res.y[:,-1] - x0) < 1e-9)

if __name__=="__main__":
    test_propagator_cr3bp(make_plot=True)
    plt.show()
