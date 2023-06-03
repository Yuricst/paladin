"""
Test for propagating NRHO in full ephemeris EOM
Baseline from: https://naif.jpl.nasa.gov/pub/naif/misc/MORE_PROJECTS/DSG/
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
spice.furnsh(os.path.join(
    "..",
    "assets",
    "spice",
    "receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp"
))  # baseline NRHO


if __name__=="__main__":
    # set epochs
    et0 = spice.utc2et("2025-01-01T08:09:36")
    tf = 60 * 86400.0  # seconds
    ets = np.linspace(et0, et0 + tf, 2000)
    naif_frame = "ECLIPJ2000"

    # extract states from SPICE
    sv_baseline = np.zeros((len(ets),6))
    for idx,et in enumerate(ets):
        sv_sc,_ = spice.spkezr("-60000", et, naif_frame, "NONE", "399")
        sv_moon, _ = spice.spkezr("301", et, naif_frame, "NONE", "399")
        sv_baseline[idx,:] = sv_sc - sv_moon
    
    # plot baseline
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection = '3d')
    ax.plot(sv_baseline[:,0], sv_baseline[:,1], sv_baseline[:,2],
        color="black", label="Baseline")
    luna2.set_equal_axis(ax, sv_baseline[:,0], sv_baseline[:,1], sv_baseline[:,2])

    # create NRHO propagator
    naif_ids = ["301", "399", "10"]
    mus = [
        4902.800118457549,
        398600.435507226,
        132712440018.0,
    ]
    prop_nbody = luna2.PropagatorNBody(
        naif_frame,
        naif_ids, 
        mus,
        lstar = 3000.0,
        use_canonical=True,
    )
    prop_nbody.summary()

    # solve IVP
    x0 = prop_nbody.dim2nondim(sv_baseline[0,:])
    tf_nondim = tf/prop_nbody.tstar
    res_nbody = prop_nbody.solve(
        et0,
        [0, tf_nondim],
        x0,
        t_eval=np.linspace(0, tf_nondim, len(ets))
    )
    print(f"x0 = \n{x0}")

    # append to plot
    ax.plot(res_nbody.y[0,:]*prop_nbody.lstar,
            res_nbody.y[1,:]*prop_nbody.lstar,
            res_nbody.y[2,:]*prop_nbody.lstar,
        color="red", label="Propagated", linewidth=0.5)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.legend()

    # error plot
    error_pos = res_nbody.y[0:3,:]*prop_nbody.lstar - np.transpose(sv_baseline[:,0:3])
    fig1, axs1 = plt.subplots(3,1, figsize=(9,5))
    for idx in range(3):
        axs1[idx].plot(ets, error_pos[idx,:])
        axs1[1].grid(True, alpha=0.5)
    axs1[0].set(xlabel="Epoch", ylabel="x error, km")
    axs1[1].set(xlabel="Epoch", ylabel="y error, km")
    axs1[2].set(xlabel="Epoch", ylabel="z error, km")
    plt.tight_layout()
    plt.show()

