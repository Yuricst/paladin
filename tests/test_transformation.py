"""
Test for transformation between spice frames
"""

import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
import os

import sys 
sys.path.append("../")
import paladin

spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "pck", "gm_de440.tpc"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "pck", "pck00011.tpc"))


if __name__=="__main__":
    # list of mus
    mus = [
        398600.44,
        4902.800066,
    ]
    naif_ids = ["399", "301"]
    et0 = spice.utc2et("2025-12-18T12:28:28")
    print(f"et0 = {et0}")
    lstar = 1.0
    tstar = 1.0
    
    # create propagator
    prop = paladin.PropagatorNBody(
        "J2000",
        naif_ids, 
        mus,
        lstar,
        tstar,
    )
    # set initial state and propagate
    x0 = np.array([
        85000.0, 0.0, 0.0,
        0.0, 0.0, np.sqrt(398600.44/85000.0),
    ])
    print(f"x0 = {x0}")
    tof = 10*86400.0
    t_eval = np.linspace(0,tof,600)
    res = prop.solve(et0, [0,tof], x0, t_eval=t_eval)
    epochs = et0 + res.t

    # convert state
    res_EJ = paladin.apply_frame_transformation(epochs, res.y, "J2000", "IAU_EARTH")
    print(res_EJ.shape)

    # transform back
    res_J = paladin.apply_frame_transformation(epochs, res_EJ, "IAU_EARTH", "J2000")

    # check
    check = res_EJ - res_J
    print(check)

    # plot figure
    fig = plt.figure(figsize = (6, 6))
    ax = plt.axes(projection = '3d')
    #fig, ax = plt.subplots(1,1,figsize=(8,6), p='3d')
    print(res.y.shape)
    ax.plot(res.y[0,:], res.y[1,:], res.y[2,:], label='J2000')
    ax.plot(res_EJ[0,:], res_EJ[1,:], res_EJ[2,:], label='ECLIPJ2000')
    ax.plot(res_J[0,:], res_J[1,:], res_J[2,:], label='J2000 (converted back)')
    plt.show()