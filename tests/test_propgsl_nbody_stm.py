"""
Test for propagating NRHO in full ephemeris EOM with STM
Baseline from: https://naif.jpl.nasa.gov/pub/naif/misc/MORE_PROJECTS/DSG/
"""

import numpy as np
import spiceypy as spice
import matplotlib
import matplotlib.pyplot as plt
import os

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import paladin

matplotlib.rcParams.update({'font.size': 14})

spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "pck", "gm_de440.tpc"))

spice.furnsh(os.path.join(os.path.dirname(__file__), "..", "assets", "spice", "earth_moon_rotating_mc.tf"))  # custom frame kernel
spice.furnsh(os.path.join(
    os.path.dirname(__file__), "..",
    "assets",
    "spice",
    "receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp"
))  # baseline NRHO


def test_propagator_nrho(verbose=False, make_plot=False):
    # set epochs
    et0 = spice.utc2et("2025-01-01T08:09:36")
    tf = 6.55 * 86400.0  # seconds
    ets = np.linspace(et0, et0 + tf, 3000)
    naif_frame = "ECLIPJ2000"

    # extract states from SPICE
    sv_baseline = np.zeros((len(ets),6))
    for idx,et in enumerate(ets):
        sv_sc,_ = spice.spkezr("-60000", et, naif_frame, "NONE", "399")
        sv_moon, _ = spice.spkezr("301", et, naif_frame, "NONE", "399")
        sv_baseline[idx,:] = sv_sc - sv_moon
    
    if make_plot:
        fig = plt.figure(figsize = (8,8))
        ax = plt.axes(projection = '3d')
        ax.plot(sv_baseline[:,0], sv_baseline[:,1], sv_baseline[:,2],
            color="black", label="Baseline")
        paladin.set_equal_axis(ax, sv_baseline[:,0], sv_baseline[:,1], sv_baseline[:,2])

    # create NRHO propagator
    naif_ids = ["301", "399", "10"]
    mus = [spice.bodvrd(ID, "GM", 1)[1][0] for ID in naif_ids]
    prop_nbody = paladin.GSLPropagatorNBody(
        naif_frame,
        naif_ids,
        mus,
        lstar = 3000.0,
        use_canonical=True,
    )
    if verbose:
        prop_nbody.summary()

    # solve IVP
    x0 = prop_nbody.dim2nondim(sv_baseline[0,:])
    tf_nondim = tf/prop_nbody.tstar
    res_nbody = prop_nbody.solve_stm(
        et0,
        [0, tf_nondim],
        x0,
        t_eval=np.linspace(0, tf_nondim, len(ets))
    )
    # print(f"Final STM: {list(res_nbody.y[6:,-1])}")
    STMf_check = [
        0.08036017968431802, 1.2021164238091913, 2.244069701918059, 
        -232.5220408603032, 88.0185300262683, 105.5554389723413, 
        0.7758967940109134, 1.00935166857871, -1.7581483950873826, 
        31.437689246974188, -94.89159869898428, -73.79858586931158, 
        0.9203641085436832, 0.1144541438711723, -1.8648021493105202, 
        58.44809223653346, -106.25746023359494, -123.76350238992137, 
        0.004155964330075827, 0.005981939971990117, 0.008476701709274899,
        -0.5446215340784266, 0.2737820103139119, 0.5211862436470267, 
        0.0047351512877628794, 0.007157034481156232, -0.018589855151601505, 
        -0.33056842263489994, 0.10890538927437082, -1.1308419363589812, 
        -0.01829141665091205, 0.0017906255832783415, 0.05308080505323328,
          -0.6370702980418691, 1.90530545746848, 3.4644777451185504]
    if make_plot:
        # append to plot
        ax.plot(res_nbody.y[0,:]*prop_nbody.lstar,
                res_nbody.y[1,:]*prop_nbody.lstar,
                res_nbody.y[2,:]*prop_nbody.lstar,
            color="red", label="Propagated", linewidth=0.5)
        ax.set(xlabel="x, km", ylabel="y, km", zlabel="z, km")
        ax.legend()
        #plt.savefig("../plots/propagation_example_nrho.png", dpi=200)

        # error plot
        error_pos = res_nbody.y[0:3,:]*prop_nbody.lstar - np.transpose(sv_baseline[:,0:3])
        fig1, axs1 = plt.subplots(3,1, figsize=(9,5))
        for idx in range(3):
            axs1[idx].plot(ets, error_pos[idx,:])
            axs1[idx].grid(True, alpha=0.5)
        axs1[0].set(xlabel="Epoch", ylabel="x error, km")
        axs1[1].set(xlabel="Epoch", ylabel="y error, km")
        axs1[2].set(xlabel="Epoch", ylabel="z error, km")
        plt.tight_layout()
    assert np.max(np.abs(res_nbody.y[6:,-1] - STMf_check)) < 1e-9


if __name__=="__main__":
    test_propagator_nrho(verbose=True, make_plot=True)
    plt.show()

