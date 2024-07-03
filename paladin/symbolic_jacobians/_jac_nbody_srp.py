"""
Symbolic computation of Jacobians with N-body+SRP EOM
"""

import sympy as sym

from ._symbolic_utils import (
    _sym_norm_length3,
    _sym_dot_length3,
    _sym_cross_prod,
    _symbolicbattin_3rd_body_perturbation,
    _symbolicj2_equator_frame_to_inertial_frame,
)


def get_jacobian_expr_nbody_srp(mu_list, cse=True):
    """Create Jacobian expression for N-body + SRP + J2 dynamics

    ODE has access to: `params = [mu_list, naif_ids, naif_frame, abcorr, et0, lstar, tstar, k_srp, k_J2]`
    
    Args:
        mu_list (list): list of gravitational parameters
        cse (bool): whether to use common subexpression elimination (default: True)

    Returns:
        (func): function to compute Jacobian
    """
    # number of bodies
    n_mus = len(mu_list)

    # Define states
    states = sym.Array(sym.symbols('state:%d' % 6))
    x,y,z,vx,vy,vz = states  # unpack
    r = sym.sqrt(x**2 + y**2 + z**2)

    # Define position vectors of 3rd bodies
    pos_3bd_list = [sym.Array([sym.symbols('state_%d_%d' % (i, j)) for j in range(3)]) for i in range(n_mus-1)]

    # Define constants for SRP
    r_sun = sym.Array(sym.symbols('r_sun_:%d' % 3))  # technically redundant but for redefined easier handling
    k_srp = sym.symbols("k_srp")

    # Define mu's
    mus = sym.symbols('mu:%d' % len(mu_list))

    # Define eoms
    mu_r3 = mus[0]/r**3
    ax = -mu_r3 * x
    ay = -mu_r3 * y
    az = -mu_r3 * z
    
    # Define third-body perturbation's
    for idx in range(n_mus-1):
        avec_3bd = _symbolicbattin_3rd_body_perturbation(
            mus[idx+1], states[0:3], pos_3bd_list[idx],
        )
        ax += avec_3bd[0]
        ay += avec_3bd[1]
        az += avec_3bd[2]

    # Define SRP
    r_relative_x = x - r_sun[0]
    r_relative_y = y - r_sun[1]
    r_relative_z = z - r_sun[2]
    r_relative_divisor = sym.sqrt(r_relative_x**2 + r_relative_y**2 + r_relative_z**2)**3
    ax += k_srp * r_relative_x / r_relative_divisor
    ay += k_srp * r_relative_y / r_relative_divisor
    az += k_srp * r_relative_z / r_relative_divisor

    # Compute sensitivities
    Uxx = sym.Matrix([
        [sym.diff(ax, x), sym.diff(ax, y), sym.diff(ax, z)],
        [sym.diff(ay, x), sym.diff(ay, y), sym.diff(ay, z)],
        [sym.diff(az, x), sym.diff(az, y), sym.diff(az, z)],
    ])

    # concatenate to form system jacobian
    zero_3by3 = sym.zeros(3,3)
    identity_3by3 = sym.eye(3)
    sensitivity_pos = zero_3by3.col_join(Uxx)
    sensitivity_vel = identity_3by3.col_join(sym.zeros(3,3))
    jac = sensitivity_pos.row_join(sensitivity_vel)

    # create function
    jac_expr = sym.utilities.lambdify(
        [states, mus, pos_3bd_list, k_srp, r_sun], 
        jac, 
        modules="numpy",
        cse = cse,
    )
    return jac_expr
