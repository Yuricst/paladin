"""
Benchmark pygsl nbody integrator
Baseline from: https://naif.jpl.nasa.gov/pub/naif/misc/MORE_PROJECTS/DSG/
"""

import numpy as np
import spiceypy as spice
import matplotlib
import matplotlib.pyplot as plt
import os
import time
from tqdm.auto import tqdm

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


def benchmark_pygsl_nbody(verbose=False, make_plot=False):
    # set epochs
    et0 = spice.utc2et("2025-01-01T08:09:36")
    tf_day = 30.0
    tf = tf_day * 86400.0  # seconds
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

    def run_solve():
        prop_nbody.solve(
            et0,
            [0, tf_nondim],
            x0,
        )
    
    def run_solve_stm():
        prop_nbody.solve_stm(
            et0,
            [0, tf_nondim],
            x0,
        )

    # test solve
    N = 5
    run_times = []
    for _ in tqdm(range(N), desc='testing solve()'):
        tstart = time.time()
        run_solve()
        run_times.append(time.time() - tstart)
    print(f"pygsl solve over {tf_day:1.2f} days (10 run samples)")
    print(f"    mean exec. time : {np.mean(run_times):.4f} s")
    if N > 1:
        print(f"    std dev. time   : {np.std(run_times):.4f} s")

    # test solve
    N = 1
    run_times = []
    for _ in tqdm(range(N), desc='testing solve_stm()'):
        tstart = time.time()
        run_solve_stm()
        run_times.append(time.time() - tstart)
    print(f"pygsl solve_stm over {tf_day:1.2f} days (10 run samples)")
    print(f"    mean exec. time : {np.mean(run_times):.4f} s")
    if N > 1:
        print(f"    std dev. time   : {np.std(run_times):.4f} s")
    return
    


if __name__=="__main__":
    benchmark_pygsl_nbody(verbose=True, make_plot=True)

