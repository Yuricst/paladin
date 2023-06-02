"""
Test for propagating NRHO in full ephemeris EOM
"""

import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
import os

import sys 
sys.path.append("../")
import luna2

spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))
spice.furnsh(os.path.join("..", "assets", "spice", "earth_moon_rotating_mc.tf"))  # custom frame kernel


if __name__=="__main__":
    # seed halo in CR3BP
    mu_cr3bp = 1.215058560962404E-2
    lstar = 389703.0
    tstar = 382981.0
    vstar = lstar/tstar

    # create N-body propagator
    et0 = 726940069.1842602
    x0 = np.array([
        -1.77628853e4,
        6.22038413e3,
        -6.89234435e4,
        8.59062825e-4,
        7.46133275e-2,
        -7.12593784e-3
    ])
    tf = 14 * 86400.0  # seconds
    mus = [
        4902.800066,
        398600.44,
    ]
    naif_frame = "ECLIPJ2000"
    prop_nbody = luna2.PropagatorNBody(
        naif_frame,
        ["301", "399"], 
        mus,
        lstar,
        tstar,
    )
    res_nbody = prop_nbody.solve(
        et0,
        [0, tf],
        x0,
        t_eval=np.linspace(0, tf, 1000)
    )

    # create N-body propagator in canonical mode
    mus_canonical = [
        1.0,
        398600.44/4902.800066,
    ]
    lstar_nbody = 3000.0
    vstar_nbody = np.sqrt(4902.800066/lstar_nbody)
    tstar_nbody = lstar_nbody/vstar_nbody
    prop_nbody_canonical = luna2.PropagatorNBody(
        naif_frame,
        ["301", "399"],
        mus_canonical,
        lstar_nbody,
        tstar_nbody,
        use_canonical=True,
    )
    res_nbody_canonical = prop_nbody_canonical.solve(
        et0,
        [0,res_nbody.t[-1]/tstar_nbody],
        np.concatenate((x0[0:3]/lstar_nbody, x0[3:6]/vstar_nbody)),
        t_eval=np.linspace(0, res_nbody.t[-1]/tstar_nbody, 1000)
    )

    # plot CR3BP trajectory
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection = '3d')
    ax.plot(res_nbody.y[0,:], res_nbody.y[1,:], res_nbody.y[2,:], label=f"N-body ({naif_frame})")
    ax.scatter(
        res_nbody.y[0,0], res_nbody.y[1,0], res_nbody.y[2,0],
        marker="o")
    
    ax.plot(res_nbody_canonical.y[0,:]*lstar_nbody,
            res_nbody_canonical.y[1,:]*lstar_nbody,
            res_nbody_canonical.y[2,:]*lstar_nbody,
        label=f"N-body ({naif_frame} - canonical)")
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.legend()
    plt.savefig("../plots/propagation_example.png", dpi=200)
    plt.show()
