"""
Test for N-body propagator
"""

import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
import os

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import paladin

spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))


def test_propagator_nbody_earth(verbose = False, make_plot = False):
    # list of mus
    mus = [
        398600.44,
        4902.800066,
    ]
    naif_ids = ["399", "301"]
    et0 = spice.utc2et("2025-12-18T12:28:28")
    
    # create propagator
    prop = paladin.PropagatorNBody(
        "J2000",
        naif_ids, 
        mus,
        lstar=10000.0,
    )
    # set initial state and propagate
    x0 = np.array([
        85000.0, 0.0, 0.0,
        0.0, 0.0, np.sqrt(398600.44/85000.0),
    ])
    tof = 10*86400.0
    t_eval = np.linspace(0,tof,600)
    res = prop.solve(et0, [0,tof], x0, t_eval=t_eval)

    xf_test = np.array([-84994.80512080838, 
                        71.26871011789203, 
                        -1752.9493905923696, 
                        0.04514412603634513, 
                        0.0014589731331479843, 
                        -2.1648758853796912])

    if make_plot:
        # plot figure
        fig = plt.figure(figsize = (6, 6))
        ax = plt.axes(projection = '3d')
        print(res.y.shape)
        ax.plot(res.y[0,:], res.y[1,:], res.y[2,:])
    assert all(np.abs(xf_test - res.y[:,-1]) < 1e-12)


if __name__=="__main__":
    test_propagator_nbody_earth(verbose = True, make_plot = True)
    plt.show()