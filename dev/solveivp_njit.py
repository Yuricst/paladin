"""Check njit usage with solve_ivp"""

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

# def lotkavolterra(t, z, a, b, c, d):
#     x, y = z
#     return [a*x - b*x*y, -c*y + d*x*y]

# @njit
# def lotkavolterra_njit(t, z, a, b, c, d):
#     x, y = z
#     return [a*x - b*x*y, -c*y + d*x*y]

# tspan = [0, 100]
# sol = solve_ivp(lotkavolterra, tspan, [10, 5],
#                 args=(1.5, 1, 3, 1))
# print(f"Final state: {list(sol.y[:,-1])}")

# sol_njit = solve_ivp(lotkavolterra_njit, tspan, [10, 5],
#                      args=(1.5, 1, 3, 1))
# print(f"Final state: {list(sol_njit.y[:,-1])}")


def twobody(t,sv,mu):
    """Two-body problem EOM"""
    r = sv[0:3]
    v = sv[3:6]
    dsv = np.zeros(6,)
    dsv[0:3] = v
    dsv[3:6] = -mu/np.linalg.norm(r)**3 * r
    return dsv


@njit
def twobody_njit(t,sv,mu):
    """Two-body problem EOM"""
    r = sv[0:3]
    v = sv[3:6]
    dsv = np.zeros(6,)
    dsv[0:3] = v
    dsv[3:6] = -mu/np.linalg.norm(r)**3 * r
    return dsv


x0 = [1, 0, 0, -0.02, 1.0, 0.04]
tspan = [0, 200 * 2*np.pi]
sol = solve_ivp(twobody, tspan, x0,
                args=(1.0,), rtol=1e-12, atol=1e-12)
print(f"Final state: {list(sol.y[:,-1])}")

sol_njit = solve_ivp(twobody_njit, tspan, x0,
                     args=(1.0,), rtol=1e-12, atol=1e-12)
print(f"Final state: {list(sol_njit.y[:,-1])}")
