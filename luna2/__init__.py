"""
Init file in module
"""

# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("numpy", "matplotlib", "numba", "scipy", "spiceypy", "pygmo")
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

# propagation
#from ._heyoka_nbody import build_taylor_nbody
from ._eom_scipy_cr3bp import eom_cr3bp
from ._eom_scipy_nbody import third_body_battin, eom_nbody
from ._propagator_cr3bp import PropagatorCR3BP
from ._propagator_nbody import PropagatorNBody

# model transition
from ._udp_FullEphemerisTransition import FullEphemerisTransition