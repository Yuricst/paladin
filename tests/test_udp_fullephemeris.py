"""
Test for full ephemeris transition
"""

import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pygmo as pg
import pygmo_plugins_nonfree as ppnf

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
    period = 1.3960732332950263E+0

    # create CR3BP propagator & propagate over one period
    prop_cr3bp = luna2.PropagatorCR3BP(mu_cr3bp)
    #t_eval = np.linspace(0,period,500)   
    t_eval = [0, period/2, period, 1.5*period, 2*period]
    res_cr3bp = prop_cr3bp.solve(
        [0,t_eval[-1]], x0_cr3bp, t_eval=t_eval, dense_output=True
    )
    states_cr3bp_MC = luna2.canonical_to_dimensional(
        luna2.shift_barycenter_to_m2(res_cr3bp.y, mu_cr3bp),
        lstar, 
        vstar
    )

    # transform to propagator's frame
    et0 = spice.utc2et("2025-12-18T12:28:28")
    epochs = et0 + res_cr3bp.t*tstar
    states_J2000 = luna2.apply_frame_transformation(
        epochs,
        states_cr3bp_MC,
        "EARTHMOONROTATINGMC",
        "J2000"
    )
    states_J2000 = luna2.dimensional_to_canonical(states_J2000, lstar, vstar)

    # create N-body propagator
    print("Creating N-body integrator...")
    et0 = spice.utc2et("2025-12-18T12:28:28")
    mus = [
        mu_cr3bp,    #4902.800066,
        1-mu_cr3bp,  #398600.44,
    ]
    prop_nbody = luna2.PropagatorNBody(
        "J2000",
        ["301", "399"], 
        mus,
        lstar,
        tstar,
        use_canonical=True,
    )
    res_nbody = prop_nbody.solve(
        et0,
        [0,res_cr3bp.t[-1]],
        states_J2000[:,0],
        t_eval=np.linspace(0,res_cr3bp.t[-1],1000)
    )

    # Construct nodes for two revolutions
    print(f"states_J2000 = {states_J2000}")
    state_half_period = states_J2000[:,1]
    nodes = [
        states_J2000[:,0],
        states_J2000[:,1],
        #states_J2000[:,2],
        #states_J2000[:,3],
        #states_J2000[:,4],
    ]
    tofs = [period/2 + idx for idx in range(len(nodes)-1)]
    tofs_bounds = [
        [tof*0.95, tof*1.05] for tof in tofs
    ]

    # create bounds on et0 and nodes
    et0_bounds = [et0 - 1000, et0 + 1000]
    nodes_bounds = [
        [states_J2000[:,0], states_J2000[:,0]],
        luna2.get_node_bounds_relative(states_J2000[:,1], [0.15, 0.05, 0.15, 0.05, 0.15, 0.05]),
        #luna2.get_node_bounds_relative(states_J2000[:,2], [0.15, 0.05, 0.15, 0.05, 0.15, 0.05]),
        #luna2.get_node_bounds_relative(states_J2000[:,3], [0.15, 0.05, 0.15, 0.05, 0.15, 0.05]),
        #luna2.get_node_bounds_relative(states_J2000[:,4], [0.15, 0.05, 0.15, 0.05, 0.15, 0.05]),
    ]

    # create UDP for full-ephemeris transition
    udp = luna2.FullEphemerisTransition(
        prop_nbody,
        et0,
        nodes,
        tofs,
        et0_bounds=et0_bounds,
        nodes_bounds=nodes_bounds,
        tofs_bounds=tofs_bounds,
    )
    lb,ub = udp.get_bounds()
    print(len(lb), len(ub))

    # test fitness function evaluation
    print("Testing fitness function evaluation...")
    xtest = (np.array(ub) + np.array(lb))/2
    fvec, sol_fwd_list, sol_bck_list = udp.fitness(xtest, True)
    print(f"fvec = {fvec}")

    # test gradient evaluation
    print("Testing gradient evaluation...")
    grad = udp.gradient(xtest)


    # plot CR3BP trajectory
    #print("Plotting initial guess...")
    #colors = cm.rainbow(np.linspace(0, 1, len(sol_fwd_list)))
    #fig = plt.figure(figsize = (6, 6))
    #ax = plt.axes(projection = '3d')
    #ax.plot(states_J2000[0,:], states_J2000[1,:], states_J2000[2,:], label="CR3BP (J2000)")
    #ax.plot(res_nbody.y[0,:], res_nbody.y[1,:], res_nbody.y[2,:], label="N-body (J2000)")

    # # plot initial arcs
    # for idx,sol in enumerate(sol_fwd_list):
    #     ax.scatter(sol.y[0,0], sol.y[1,0], sol.y[2,0], label="Fwd segments", c=colors[idx], marker="o")
    #     ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], label="Fwd segments", c=colors[idx])
    # for idx,sol in enumerate(sol_bck_list):
    #     ax.scatter(sol.y[0,0], sol.y[1,0], sol.y[2,0], label="Fwd segments", c=colors[idx], marker="s")
    #     ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], label="Bck segments", c=colors[idx])
    # #ax.legend()


    # solve with NLP solver
    print("Creating algorithm...")
    algo = luna2.algo_gradient(
        name="ipopt", 
        #snopt7_path=os.getenv("SNOPT_SO"),
        #max_iter=5, ctol=1e-5
    )
    
    algo.set_verbosity(1)
    prob = pg.problem(udp)
    pop = pg.population(prob, 0)
    pop.push_back(xtest)
    print("len(pop): ", len(pop))

    print("Solving...")
    pop = algo.evolve(pop)

    # get optimal solution
    xopt = pop.champion_x
    fvec, sol_fwd_list, sol_bck_list = udp.fitness(xopt, True)
    print("fvec = ", fvec)

    # plot optimized arcs
    colors = cm.rainbow(np.linspace(0, 1, len(sol_fwd_list)))
    fig = plt.figure(figsize = (6, 6))
    ax = plt.axes(projection = '3d')
    for idx,sol in enumerate(sol_fwd_list):
        ax.scatter(sol.y[0,0], sol.y[1,0], sol.y[2,0], label="Fwd segments", color=colors[idx], marker="o")
        ax.scatter(sol.y[0,-1], sol.y[1,-1], sol.y[2,-1], label="Fwd segments", color=colors[idx], marker="^")
        ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], label="Fwd segments", color=colors[idx])
    for idx,sol in enumerate(sol_bck_list):
        ax.scatter(sol.y[0,0], sol.y[1,0], sol.y[2,0], label="Fwd segments", color=colors[idx], marker="s")
        ax.scatter(sol.y[0,-1], sol.y[1,-1], sol.y[2,-1], label="Fwd segments", color=colors[idx], marker="v")
        ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], label="Bck segments", color=colors[idx])
  

    # display plots
    plt.show()