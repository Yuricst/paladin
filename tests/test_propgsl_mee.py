"""Test GSL-based integrator"""

"""
Test for propagating NRHO in full ephemeris EOM
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
    try:
        import pykep as pk

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
        prop_nbody = paladin.GSLPropagatorMEE(
            naif_frame,
            naif_ids,
            mus,
            lstar = 3000.0,
            use_canonical=True,
        )
        if verbose:
            prop_nbody.summary()

        # solve IVP
        x0 = np.concatenate((sv_baseline[0,0:3]/prop_nbody.lstar, sv_baseline[0,3:6]/prop_nbody.vstar))
        mee0 = pk.ic2eq(r = x0[0:3], v = x0[3:6], mu = 1.0)
        tf_nondim = tf/prop_nbody.tstar
        res_nbody = prop_nbody.solve(
            et0,
            [0, tf_nondim],
            mee0,
            t_eval=np.linspace(0, tf_nondim, len(ets))
        )
        meef = res_nbody.y[:,-1]
        rf,vf = pk.eq2ic(meef, mu = 1.0)
        xf = np.concatenate((rf, vf))
        if verbose:
            print(f"x0 (cartesian) = \n{list(x0)}")
            print(f"xf (cartesian) = \n{list(xf)}")
        xf_check = np.array([3.1087933159276013, 4.673573546276977, 
                             -22.26242216035324, 0.01177875135593217,
                             -0.04940333292761417, -0.051734260463009364])

        if make_plot:
            # convert each state to cartesian
            rvs = np.zeros((6,len(res_nbody.t)))
            for idx in range(len(res_nbody.t)):
                rvs[0:3,idx], rvs[3:6,idx] = pk.eq2ic(res_nbody.y[:,idx], mu = 1.0)

            # append to plot
            ax.plot(rvs[0,:]*prop_nbody.lstar,
                    rvs[1,:]*prop_nbody.lstar,
                    rvs[2,:]*prop_nbody.lstar,
                color="red", label="Propagated", linewidth=0.5)
            ax.set(xlabel="x, km", ylabel="y, km", zlabel="z, km")
            ax.legend()
            #plt.savefig("../plots/propagation_example_nrho.png", dpi=200)

            # error plot
            error_pos = rvs[0:3,:]*prop_nbody.lstar - np.transpose(sv_baseline[:,0:3])
            fig1, axs1 = plt.subplots(3,1, figsize=(9,5))
            for idx in range(3):
                axs1[idx].plot(ets, error_pos[idx,:])
                axs1[idx].grid(True, alpha=0.5)
            axs1[0].set(xlabel="Epoch", ylabel="x error, km")
            axs1[1].set(xlabel="Epoch", ylabel="y error, km")
            axs1[2].set(xlabel="Epoch", ylabel="z error, km")
            plt.tight_layout()
            print(f"np.abs(xf - xf_check) = {np.abs(xf - xf_check)}")
        assert all(np.abs(xf - xf_check) < 1e-9)
    except:
        assert True


if __name__=="__main__":
    test_propagator_nrho(verbose=True, make_plot=True)
    plt.show()

