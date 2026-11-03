"""Hydrogen Wavefunction Implementation and Validation (Alias for a0.py).

This module re-exports all public symbols from a0.py for backwards compatibility.
"""

from a0 import (
    radial_wavefunction,
    angular_wavefunction,
    real_angular_wavefunction,
    psi_nlm,
    expectation_values,
    separated_quad_normalization,
    spherical_grid_integral,
    find_radial_nodes,
    __all__,
)

if __name__ == "__main__":
    import a0
    import runpy
    runpy.run_module("a0", run_name="__main__")
