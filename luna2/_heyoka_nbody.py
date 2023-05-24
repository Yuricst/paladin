"""
Heyoka implementation of N-body problem
"""

import heyoka as hy
import numpy as np
import spiceypy as spice


def third_body_battin(r, s, mu_third):
    """Third body perturbation acceleration via Battin's formulation"""
    d = r - s
    dnorm = hy.sqrt(np.dot(d,d))
    q = np.dot(r, r - 2*s)/np.dot(s,s)
    F = q*((3 + 3*q + q**2)/(1 + hy.sqrt(1+q)**3))
    return -mu_third/dnorm**3 * (r + F*s)


def build_taylor_nbody(mus, naif_ids, et0, lstar=1.0, tstar=1.0):
    # Create the symbolic variables.
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    # set hy.time to 0 for spkpos not to break
    if isinstance(hy.par[0] + hy.time, float):
        epoch = hy.par[0] + hy.time/tstar
    else:
        epoch = et0
    print(f"epoch = {epoch}")

    # Compute accelerations
    dvxdt = -mus[0]*x/((x**2+y**2+z**2)**(3/2))
    dvydt = -mus[0]*x/((x**2+y**2+z**2)**(3/2))
    dvzdt = -mus[0]*x/((x**2+y**2+z**2)**(3/2))

    # Iterate through extra bodies
    for idx in range(len(mus)-1):
        rvec3, _ = spice.spkpos(
            naif_ids[idx+1], epoch, "J2000", "NONE", naif_ids[0]
        )
        rvec3 /= lstar
        accel3 = third_body_battin(np.array([x,y,z]), rvec3, mus[idx+1])
        # Add to accelerations
        dvxdt += accel3[0]
        dvydt += accel3[1]
        dvzdt += accel3[2]

    # Create Taylor integrator
    ode_sys  = [
        (x, vx),
        (y, vy),
        (z, vz),
        (vx, dvxdt),
        (vy, dvydt),
        (vz, dvzdt),
    ]
    tmp_ic = [1., 0., 0., 0., 1., 0.]
    return hy.taylor_adaptive(ode_sys, tmp_ic, pars=[et0,])


if __name__=="__main__":
    # list of mus
    mus = [
        398600.44,
        4902.800066,
    ]
    naif_ids = ["399", "301"]
    et0 = spice.utc2et("2025-12-18T12:28:28")
    print(f"et0 = {et0}")
    # build ta
    ta = build_taylor_nbody(mus, naif_ids, et0)