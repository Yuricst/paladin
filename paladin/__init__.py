"""
Init file in module
"""

__copyright__    = 'Copyright (C) 2023 Yuri Shimane'
__version__      = '0.1.3'
__license__      = 'MIT License'
__author__       = 'Yuri Shimane'
__author_email__ = 'yuri.shimane@gatech.edu'
__url__          = 'https://github.com/Yuricst/paladin'


# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("numpy", "matplotlib", "scipy", "spiceypy", "sympy", "pygsl")
_missing_dependencies = []

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _hard_dependencies, _dependency, _missing_dependencies


# misc
from ._transformation import (
    shift_origin_x,
    shift_barycenter_to_m2,
    apply_frame_transformation,
    canonical_to_dimensional,
    dimensional_to_canonical,
)
from ._nodes_helper import (
    get_node_bounds_relative,
)
from ._plotter import (
    set_equal_axis
)
from ._newtonraphson import (
    _newtonraphson_iteration,
    _leastsquare_iteration,
    _minimumnorm_iteration,
)

# symbolic jacobian
from .symbolic_jacobians._jac_nbody import get_jacobian_expr_nbody
from .symbolic_jacobians._jac_nbody_srp import get_jacobian_expr_nbody_srp

# equations of motion
from ._perturbations import third_body_battin
from .eoms._eom_cr3bp import eom_cr3bp
from .eoms._eom_nbody import third_body_battin, eom_nbody, eomstm_nbody
from .eoms._eom_nbody_srp import eom_nbody_srp, eomstm_nbody_srp
from .eoms._eom_mee import mee2rv, eom_mee

# propagators
from ._propagator_scipy_cr3bp import PropagatorCR3BP
from ._propagator_scipy_nbody import PropagatorNBody
from ._propagator_gsl_mee import GSLPropagatorMEE
from ._wrap_propagator import PropagatorWrapper

from ._propagator_gsl_nbody import GSLPropagatorNBody

#try:
#     from ._propagator_gsl_nbody import GSLPropagatorNBody
#     ext_gsl = True
# except:
#     print(f"WARNING : skipping GSL-dependent functions")
#     ext_gsl = False
#     pass

try:
    # model transition
    from ._udp_FullEphemerisTransition import FullEphemerisTransition

    # algorithms for pygmo
    from ._algo_factory import algo_gradient
    ext_pygmo = True
except:
    print(f"WARNING : skipping pygmo-dependent functions")
    ext_pygmo = False
    pass