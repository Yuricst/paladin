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


if __name__=="__main__":
    et0 = spice.utc2et("2025-12-18T12:28:28")
    prob = luna2.FullEphemerisTransition()

    # seed halo in CR3BP
    mu_cr3bp = 1.215058560962404E-2
    lstar = 389703.0
    tstar = 382981.0

    # ID723 in southern halo
    x0 = np.array([
        1.0134037728554581E+0, 0, -1.7536227281091840E-1,
        0, -8.3688427472776439E-2, 0
    ])
    period = 1.3960732332950263E+0

    # create CR3BP propagator
    prop_cr3bp = luna2.PropagatorCR3BP(mu_cr3bp)

    # create N-body propagator
    mus = [
        398600.44,
        4902.800066,
    ]
    prop_nbody = luna2.PropagatorNBody(
        "J2000",
        ["399", "301"], 
        mus,
        lstar,
        tstar,
    )