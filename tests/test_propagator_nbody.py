"""
Test for full ephemeris transition
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

    # ID723 in southern halo
    x0_cr3bp = np.array([
        1.0134037728554581E+0, 0, -1.7536227281091840E-1,
        0, -8.3688427472776439E-2, 0
    ])
    period = 2 * 1.3960732332950263E+0

    # create CR3BP propagator & propagate over one period
    prop_cr3bp = luna2.PropagatorCR3BP(mu_cr3bp)
    res_cr3bp = prop_cr3bp.solve([0,period], x0_cr3bp, t_eval=np.linspace(0,period,1000))
    states_cr3bp_MC = luna2.canonical_to_dimensional(
        luna2.shift_barycenter_to_m2(res_cr3bp.y, mu_cr3bp),
        lstar, 
        vstar
    )

    # transform to propagator's state
    naif_frame = "ECLIPJ2000"
    et0 = spice.utc2et("2025-12-18T12:28:28")
    epochs = et0 + res_cr3bp.t*tstar
    states_J2000 = luna2.apply_frame_transformation(
        epochs,
        states_cr3bp_MC,
        "EARTHMOONROTATINGMC",
        naif_frame
    )

    # create N-body propagator
    et0 = spice.utc2et("2025-12-18T12:28:28")
    mus = [
        4902.800066,
        398600.44,
    ]
    prop_nbody = luna2.PropagatorNBody(
        naif_frame,
        ["301", "399"], 
        mus,
        lstar,
        tstar,
    )
    res_nbody = prop_nbody.solve(
        et0,
        [0,res_cr3bp.t[-1]*tstar],
        states_J2000[:,0],
        t_eval=np.linspace(0,res_cr3bp.t[-1]*tstar,1000)
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
        luna2.dimensional_to_canonical(states_J2000, lstar_nbody, vstar_nbody)[:,0],
        t_eval=np.linspace(0, res_nbody.t[-1]/tstar_nbody, 1000)
    )

    # plot CR3BP trajectory
    fig = plt.figure(figsize = (6, 6))
    ax = plt.axes(projection = '3d')
    ax.plot(states_cr3bp_MC[0,:], states_cr3bp_MC[1,:], states_cr3bp_MC[2,:], label="CR3BP (rot-MC)")
    ax.plot(states_J2000[0,:], states_J2000[1,:], states_J2000[2,:], label=f"CR3BP ({naif_frame})")
    #ax.plot(_states_cr3bp_MC[0,:], _states_cr3bp_MC[1,:], _states_cr3bp_MC[2,:], label="CR3BP")
    ax.plot(res_nbody.y[0,:], res_nbody.y[1,:], res_nbody.y[2,:], label=f"N-body ({naif_frame})")


    ax.plot(res_nbody_canonical.y[0,:]*lstar_nbody,
            res_nbody_canonical.y[1,:]*lstar_nbody,
            res_nbody_canonical.y[2,:]*lstar_nbody,
        label=f"N-body ({naif_frame} - canonical)")
    
    ax.legend()
    plt.savefig("../plots/propagation_example.png", dpi=200)
    plt.show()
