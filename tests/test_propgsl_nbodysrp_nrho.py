"""
Test for propagating NRHO in full ephemeris EOM + SRP
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
    tf = 60 * 86400.0  # seconds
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
    B_srp = 0.01
    prop_nbody = paladin.GSLPropagatorNBody(
        naif_frame,
        naif_ids,
        mus,
        lstar = 3000.0,
        B_srp = B_srp,
        use_canonical=True,
        use_srp = True,
    )
    if verbose:
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
    if verbose:
        print(f"x0 = \n{x0}")
        print(f"xf = \n{list(res_nbody.y[:,-1])}")
    xf_check = np.array([ 3.8587058893623105, 
                          4.449609685025795,
                         -22.368115789137285,
                          0.014778573532166881, 
                         -0.0469002925530423, 
                         -0.049388608994282164])

    if make_plot:
        # append to plot
        ax.plot(res_nbody.y[0,:]*prop_nbody.lstar,
                res_nbody.y[1,:]*prop_nbody.lstar,
                res_nbody.y[2,:]*prop_nbody.lstar,
            color="red", label="Propagated", linewidth=0.5)
        ax.set(xlabel="x, km", ylabel="y, km", zlabel="z, km")
        ax.legend()
        plt.savefig("../plots/propagation_example_nrho.png", dpi=200)

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
    assert all(np.abs(res_nbody.y[:,-1] - xf_check) < 1e-9)


if __name__=="__main__":
    test_propagator_nrho(verbose=True, make_plot=True)
    plt.show()

