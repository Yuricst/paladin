"""
Test for propagating NRHO in full ephemeris EOM + SRP with STM
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
    res_nbody = prop_nbody.solve_stm(
        et0,
        [0, tf_nondim],
        x0,
        t_eval=np.linspace(0, tf_nondim, len(ets))
    )
    print(f"Final STM: {list(res_nbody.y[6:,-1])}")
    STMf_check = np.array([
        0.08044333187437325, 1.2021340222341372, 2.2439252869393314, 
        -232.5238554776228, 88.01689342356566, 105.5468532366416, 
        0.7758451034534123, 1.0094069966277923, -1.7577900246033633, 
        31.435514192616637, -94.8844928435299, -73.7777384334029, 
        0.9200859919826699, 0.11446426285539384, -1.863999476442602, 
        58.41986410811472, -106.21624712629792, -123.72058834730434, 
        0.004156850515021149, 0.005981739159739276, 0.00847348556560287, 
        -0.5445540262988118, 0.2736726194090909, 0.5210028637893412, 
        0.004731813502206093, 0.007156952637076679, -0.018579401842186396, 
        -0.3307787093595196, 0.10932105918420555, -1.1302220349125733, 
        -0.01828905370802891, 0.0017890679361536076, 0.05307549175509512, 
        -0.6367069741744407, 1.904887871547578, 3.464174577724569])
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
    assert np.max(np.abs(res_nbody.y[6:,-1] - STMf_check)) < 1e-14


if __name__=="__main__":
    test_propagator_nrho(verbose=True, make_plot=True)
    plt.show()

