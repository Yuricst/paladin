"""
Init file in module
"""

# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("numpy", "matplotlib", "scipy", "spiceypy", "sympy")
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
from ._symbolic_jacobians import get_jaocbian_expr_Nbody, get_jaocbian_expr_Nbody_srp_j2

# propagation
from ._eom_scipy_cr3bp import eom_cr3bp
from ._eom_scipy_nbody import third_body_battin, eom_nbody
from ._propagator_scipy_cr3bp import PropagatorCR3BP
from ._propagator_scipy_nbody import PropagatorNBody
from ._wrap_propagator import PropagatorWrapper

try:
    from ._propagator_gsl_nbody import GSLPropagatorNBody
except:
    print(f"WARNING : skipping GSL-dependent functions")
    pass

try:
    # model transition
    from ._udp_FullEphemerisTransition import FullEphemerisTransition

    # algorithms for pygmo
    from ._algo_factory import algo_gradient
except:
    print(f"WARNING : skipping pygmo-dependent functions")
    pass