"""
Symbolic computation of Jacobians
"""

import sympy as sym


def _sym_norm_length3(sym_vec):
    """Symbolic norm alias for length-3 vector"""
    return sym.sqrt(sym_vec[0]**2 + sym_vec[1]**2 + sym_vec[2]**2)


def _sym_dot_length3(sym_vec0, sym_vec1):
    """Symbolic dot product alias for length-3 vector"""
    return sym_vec0[0]*sym_vec1[0] + sym_vec0[1]*sym_vec1[1] + sym_vec0[2]*sym_vec1[2]


def _sym_cross_prod(sym_vec0, sym_vec1):
    return sym.Array([
        sym_vec0[1]*sym_vec1[2] - sym_vec0[2]*sym_vec1[1],
        sym_vec0[2]*sym_vec1[0] - sym_vec0[0]*sym_vec1[2],
        sym_vec0[0]*sym_vec1[1] - sym_vec0[1]*sym_vec1[0],
    ])

def _symbolicbattin_3rd_body_perturbation(mu, rvec, svec):
    """Symbolic version of Battin's formula for computing third-body perturbation acceleration"""
    # relative vector
    dvec = rvec - svec
    # q (scalar)
    q = _sym_dot_length3(rvec, rvec - 2*svec)/_sym_dot_length3(svec, svec)
    # Battin's F(q) function
    F = q*((3 + 3*q + q**2)/(1 + sym.sqrt(1 + q)**3))
    return -mu/_sym_norm_length3(dvec)**3 * (rvec + F*svec)


def _symbolicj2_equator_frame_to_inertial_frame(rvec, mu_r3, k_j2, T_IE):
    """Compute acceleration due to J2 in equator-based frame and convert to inertial frame"""
    # convert position vector from inertial to equator frame
    rvec_spacecraft_equator = T_IE.T * rvec  #np.dot(T_IE.T, rvec)
    r = _sym_norm_length3(rvec_spacecraft_equator)
    # compute constants
    k_j2_re2 = k_j2 / r**2
    z2_r2 = (rvec_spacecraft_equator[2] / r)**2
    # copmute acceleration (in equator frame)
    a_j2_equator = -mu_r3 * sym.Matrix([
        rvec_spacecraft_equator[0]*k_j2_re2 * (1 - 5*z2_r2),
        rvec_spacecraft_equator[1]*k_j2_re2 * (1 - 5*z2_r2),
        rvec_spacecraft_equator[2]*k_j2_re2 * (3 - 5*z2_r2),
    ])
    # convert acceleration from equator to inertial frame
    return T_IE * a_j2_equator   #np.dot(T_IE, a_j2_equator)

