"""
Multiple shooting with full-ephemeris problem
"""


import numpy as np
from scipy.linalg import block_diag
import spiceypy as spice
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time

import sys 
sys.path.append("../")
import luna2


matplotlib.rcParams.update({'font.size': 14})

spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "pck", "gm_de440.tpc"))

spice.furnsh(os.path.join("..", "assets", "spice", "earth_moon_rotating_mc.tf"))  # custom frame kernel


def plot_time_history(sol_fwd_list, sol_bck_list, et0_fwd_list, et0_bck_list, prop_nbody):
    fig, axs = plt.subplots(2,3,figsize = (9, 6))
    for i_sol,sol0 in enumerate(sol_fwd_list):
        for idx in range(3):
            # epoch
            et0 = et0_fwd_list[i_sol]
            # positions
            axs[0,idx].scatter(et0+sol0.t[0]*prop_nbody.tstar, sol0.y[idx,0],
                marker="o", color="dodgerblue", linewidth=0.5)
            axs[0,idx].plot(et0+sol0.t*prop_nbody.tstar, sol0.y[idx,:],
                color="dodgerblue", linewidth=0.5)
            axs[0,idx].scatter(et0+sol0.t[-1]*prop_nbody.tstar, sol0.y[idx,-1],
                marker="s", edgecolors="dodgerblue", facecolors="None", linewidth=0.5)
            # velocities
            axs[1,idx].scatter(et0+sol0.t[0]*prop_nbody.tstar, sol0.y[idx+3,0],
                marker="o", color="dodgerblue", linewidth=0.5)
            axs[1,idx].plot(et0+sol0.t*prop_nbody.tstar, sol0.y[idx+3,:],
                color="dodgerblue", linewidth=0.5)
            axs[1,idx].scatter(et0+sol0.t[-1]*prop_nbody.tstar, sol0.y[idx+3,-1],
                marker="s", edgecolors="dodgerblue", facecolors="None", linewidth=0.5)
    
    for i_sol,sol0 in enumerate(sol_bck_list):
        for idx in range(3):
            # epoch
            et0 = et0_bck_list[i_sol]
            # positions
            axs[0,idx].scatter(et0+sol0.t[0]*prop_nbody.tstar, sol0.y[idx,0],
                marker="o", color="magenta", linewidth=0.5)
            axs[0,idx].plot(et0+sol0.t*prop_nbody.tstar, sol0.y[idx,:],
                color="magenta", linewidth=0.5)
            axs[0,idx].scatter(et0+sol0.t[-1]*prop_nbody.tstar, sol0.y[idx,-1],
                marker="s", edgecolors="magenta", facecolors="None", linewidth=0.5)
            # velocities
            axs[1,idx].scatter(et0+sol0.t[0]*prop_nbody.tstar, sol0.y[idx+3,0],
                marker="o", color="magenta", linewidth=0.5)
            axs[1,idx].plot(et0+sol0.t*prop_nbody.tstar, sol0.y[idx+3,:],
                color="magenta", linewidth=0.5)
            axs[1,idx].scatter(et0+sol0.t[-1]*prop_nbody.tstar, sol0.y[idx+3,-1],
                marker="s", edgecolors="magenta", facecolors="None", linewidth=0.5)
    axs[0,0].set(xlabel="t", ylabel="x, LU")
    axs[0,1].set(xlabel="t", ylabel="y, LU")
    axs[0,2].set(xlabel="t", ylabel="z, LU")
    axs[1,0].set(xlabel="t", ylabel="vx, LU/TU")
    axs[1,1].set(xlabel="t", ylabel="vy, LU/TU")
    axs[1,2].set(xlabel="t", ylabel="vz, LU/TU")
    plt.tight_layout()
    return fig, axs

def test_multiple_shooting():
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
    period = 1.3960732332950263E+0

    # create CR3BP propagator & propagate over one period
    prop_cr3bp = luna2.PropagatorCR3BP(mu_cr3bp)
    #t_eval = np.linspace(0,period,500)
    node_interval = period/3
    n_node = 15
    t_eval = [node_interval*i for i in range(n_node)]
    res_cr3bp = prop_cr3bp.solve(
        [0,t_eval[-1]], x0_cr3bp, t_eval=t_eval, dense_output=True
    )
    states_cr3bp_MC = luna2.canonical_to_dimensional(
        luna2.shift_barycenter_to_m2(res_cr3bp.y, mu_cr3bp),
        lstar, 
        vstar
    )

    # transform CR3BP guess to propagator's frame
    naif_frame = "ECLIPJ2000"
    et0_str = "2025-12-24T18:30:28"  #"2025-12-18T12:28:28"
    print(f"et0 = {et0_str}")
    et0 = spice.utc2et(et0_str)
    
    epochs = et0 + res_cr3bp.t*tstar
    states_J2000 = luna2.apply_frame_transformation(
        epochs,
        states_cr3bp_MC,
        "EARTHMOONROTATINGMC",
        naif_frame
    )

    # create high-fidelity propagator
    print("Creating N-body integrator...")
    lstar_nbody = 3000.0
    naif_ids = ["301", "399", "10"]
    mus = [spice.bodvrd(ID, "GM", 1)[1][0] for ID in naif_ids]

    prop_nbody = luna2.PropagatorNBody(
        naif_frame,
        naif_ids, 
        mus,
        lstar_nbody,
        use_canonical=True,
    )
    tf_nbody = res_cr3bp.t[-1]*tstar/prop_nbody.tstar
    states_J2000 = luna2.dimensional_to_canonical(states_J2000, prop_nbody.lstar, prop_nbody.vstar)
    prop_nbody.summary()

    # integrate
    res_nbody = prop_nbody.solve(
        et0,
        [0, tf_nbody],
        states_J2000[:,0],
        t_eval=np.linspace(0,tf_nbody,1000)
    )

    # plot CR3BP trajectory
    # fig = plt.figure(figsize = (6, 6))
    # ax = plt.axes(projection = '3d')
    # ax.plot(res_nbody.y[0,:]*lstar_nbody,
    #         res_nbody.y[1,:]*lstar_nbody,
    #         res_nbody.y[2,:]*lstar_nbody,
    #     label=f"N-body ({naif_frame} - canonical)")

    # Construct nodes for two revolutions
    nodes = [states_J2000[:,i] for i in range(n_node)]
    tofs = [node_interval*tstar/prop_nbody.tstar for i in range(n_node-1)]
    #[period*tstar/prop_nbody.tstar/2 + idx for idx in range(len(nodes)-1)]
    tofs_bounds = [
        [tof*0.85, tof*1.15] for tof in tofs
    ]

    # create bounds on et0 and nodes
    delta_et0_bounds = [-1000, 1000]
    nodes_bounds = [
        luna2.get_node_bounds_relative(states_J2000[:,i], [0.15, 0.05, 0.15, 0.05, 0.15, 0.05])
        for i in range(n_node)
    ]

    # create UDP for full-ephemeris transition
    udp = luna2.FullEphemerisTransition(
        prop_nbody,
        et0,
        nodes,
        tofs,
        delta_et0_bounds=delta_et0_bounds,
        nodes_bounds=nodes_bounds,
        tofs_bounds=tofs_bounds,
        economic_gradient=True,
    )
    udp.summary()

    # solve multiple shooting problem
    x0 = udp.get_x0()
    _, sol_fwd_list, sol_bck_list, et0_fwd_list, et0_bck_list = udp.fitness(x0, get_sols=True, verbose=False)

    # plot before diff-corr
    fig1 = plt.figure(figsize = (6, 6))
    ax = plt.axes(projection = '3d')
    for sol0 in sol_fwd_list:
        ax.scatter(sol0.y[0,0]*prop_nbody.lstar, sol0.y[1,0]*prop_nbody.lstar, sol0.y[2,0]*prop_nbody.lstar,
            marker="o", color="dodgerblue", linewidth=0.5)
        ax.plot(sol0.y[0,:]*prop_nbody.lstar, sol0.y[1,:]*prop_nbody.lstar, sol0.y[2,:]*prop_nbody.lstar,
            color="dodgerblue", linewidth=0.5)
        ax.scatter(sol0.y[0,-1]*prop_nbody.lstar, sol0.y[1,-1]*prop_nbody.lstar, sol0.y[2,-1]*prop_nbody.lstar,
            marker="s", edgecolors="dodgerblue", facecolors="None", linewidth=0.5)

    for sol1 in sol_bck_list:
        ax.scatter(sol1.y[0,0]*prop_nbody.lstar, sol1.y[1,0]*prop_nbody.lstar, sol1.y[2,0]*prop_nbody.lstar,
            marker="o", color="magenta", linewidth=0.5)
        ax.plot(sol1.y[0,:]*prop_nbody.lstar, sol1.y[1,:]*prop_nbody.lstar, sol1.y[2,:]*prop_nbody.lstar,
            color="magenta", linewidth=0.5)
        ax.scatter(sol1.y[0,-1]*prop_nbody.lstar, sol1.y[1,-1]*prop_nbody.lstar, sol1.y[2,-1]*prop_nbody.lstar,
            marker="d", edgecolors="magenta", facecolors="None", linewidth=0.5)
    ax.set_title("Initial guess")
    ax.set(xlabel="x, km", ylabel="y, km", zlabel="z, km")

    # time-history
    fig1b, axs = plot_time_history(sol_fwd_list, sol_bck_list, et0_fwd_list, et0_bck_list, prop_nbody)
    

    # run multiple shooting
    tstart = time.time()
    xs_list, fs_list, convergence_flag = udp.multiple_shooting(
        x0, max_iter=20, damping=1.0,
    )
    tend = time.time()
    print(f"Elapsed time: {tend-tstart:1.4f} sec; convergence_flag = {convergence_flag}")
    _, sol_fwd_list, sol_bck_list, et0_fwd_list, et0_bck_list = udp.fitness(xs_list[-1], get_sols=True, verbose=False)

    # plot after diff-corr
    fig2 = plt.figure(figsize = (6, 6))
    ax = plt.axes(projection = '3d')
    for sol0 in sol_fwd_list:
        ax.scatter(sol0.y[0,0]*prop_nbody.lstar, sol0.y[1,0]*prop_nbody.lstar, sol0.y[2,0]*prop_nbody.lstar,
            marker="o", color="dodgerblue", linewidth=0.5)
        ax.plot(sol0.y[0,:]*prop_nbody.lstar, sol0.y[1,:]*prop_nbody.lstar, sol0.y[2,:]*prop_nbody.lstar,
            color="dodgerblue", linewidth=0.5)
        ax.scatter(sol0.y[0,-1]*prop_nbody.lstar, sol0.y[1,-1]*prop_nbody.lstar, sol0.y[2,-1]*prop_nbody.lstar,
            marker="s", edgecolors="dodgerblue", facecolors="None", linewidth=0.5)

    for sol1 in sol_bck_list:
        ax.scatter(sol1.y[0,0]*prop_nbody.lstar, sol1.y[1,0]*prop_nbody.lstar, sol1.y[2,0]*prop_nbody.lstar,
            marker="o", color="magenta", linewidth=0.5)
        ax.plot(sol1.y[0,:]*prop_nbody.lstar, sol1.y[1,:]*prop_nbody.lstar, sol1.y[2,:]*prop_nbody.lstar,
            color="magenta", linewidth=0.5)
        ax.scatter(sol1.y[0,-1]*prop_nbody.lstar, sol1.y[1,-1]*prop_nbody.lstar, sol1.y[2,-1]*prop_nbody.lstar,
            marker="d", edgecolors="magenta", facecolors="None", linewidth=0.5)
    ax.set_title("Multiple shooting result")
    ax.set(xlabel="x, km", ylabel="y, km", zlabel="z, km")
    plt.savefig("../plots/test_multiple_shooting.png", dpi=300)
    #luna2.set_equal_axis(ax, sol0.y[0,:]*prop_nbody.lstar, sol0.y[1,:]*prop_nbody.lstar, sol0.y[2,:]*prop_nbody.lstar,
    #    scale=1.25, dim3=True)

    # plot time-history
    fig2b, axs = plot_time_history(sol_fwd_list, sol_bck_list, et0_fwd_list, et0_bck_list, prop_nbody)
    return


if __name__=="__main__":
    test_multiple_shooting()
    plt.show()