"""Hydrogen Wavefunction Implementation and Validation.

This module provides exact analytical implementations of the hydrogen atom
radial, angular, and total wavefunctions in atomic units (Bohr radius a0 = 1).
It uses SciPy's associated Laguerre polynomials (genlaguerre) and spherical
harmonics (sph_harm_y) with caching and high-precision integration routines.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy.integrate
import scipy.optimize
from scipy.special import genlaguerre, sph_harm_y

__all__ = [
    "radial_wavefunction",
    "angular_wavefunction",
    "real_angular_wavefunction",
    "psi_nlm",
    "expectation_values",
    "separated_quad_normalization",
    "spherical_grid_integral",
    "find_radial_nodes",
]


@functools.lru_cache(maxsize=128)
def _get_radial_cache(n: int, l: int) -> Tuple[float, Any]:
    """Compute and cache the normalization constant and Laguerre polynomial for (n, l).

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Orbital quantum number (0 <= l < n).

    Returns
    -------
    norm : float
        Analytical normalization prefactor sqrt((2/n)^3 * (n-l-1)! / (2n * (n+l)!)).
    laguerre_poly : scipy.special.orthopoly1d
        Associated Laguerre polynomial L_{n-l-1}^{2l+1}.
    """
    norm = np.sqrt(
        (2.0 / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)
    return norm, laguerre_poly


def radial_wavefunction(n: int, l: int, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute the radial wavefunction R_nl(r) of hydrogen in atomic units (a0 = 1).

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Azimuthal (orbital) quantum number (0 <= l < n).
    r : float or numpy.ndarray
        Radial distance from the nucleus in atomic units (Bohr radii). Must be non-negative.

    Returns
    -------
    float or numpy.ndarray
        Radial wavefunction value(s) R_nl(r). Returns a float if r is a scalar,
        or an ndarray if r is an array (including size-1 arrays).

    Raises
    ------
    ValueError
        If quantum numbers or radial coordinates are invalid, negative, or non-finite.
    """
    if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n >= 1):
        raise ValueError(f"Principal quantum number n must be an integer >= 1 (got {n}).")
    if not (isinstance(l, (int, np.integer)) and not isinstance(l, bool) and 0 <= l < n):
        raise ValueError(f"Orbital quantum number l must satisfy 0 <= l < n (got l={l}, n={n}).")

    r_arr = np.asarray(r, dtype=float)

    if not np.all(np.isfinite(r_arr)):
        raise ValueError("Radial coordinate r must contain only finite numbers.")
    if np.any(r_arr < 0.0):
        raise ValueError("Radial coordinate r must be non-negative (r >= 0).")

    norm, laguerre_poly = _get_radial_cache(int(n), int(l))

    rho = (2.0 * r_arr) / n
    r_nl = norm * np.exp(-rho / 2.0) * (rho ** l) * laguerre_poly(rho)

    if np.ndim(r) == 0:
        return float(r_nl.item())
    return r_nl


def angular_wavefunction(l: int, m: int, theta: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
    """Compute the angular wavefunction Y_l^m(theta, phi) using spherical harmonics.

    Parameters
    ----------
    l : int
        Orbital quantum number (l >= 0).
    m : int
        Magnetic quantum number (-l <= m <= l).
    theta : float or numpy.ndarray
        Polar (colatitudinal) angle in radians, canonical range [0, pi].
    phi : float or numpy.ndarray
        Azimuthal angle in radians, canonical range [0, 2*pi].

    Returns
    -------
    complex or numpy.ndarray
        Spherical harmonic Y_l^m(theta, phi). Note that Y_l^m is generally complex-valued
        for m != 0 and is unitless.

    Notes
    -----
    Range contract: While canonical spherical coordinates use theta in [0, pi] and
    phi in [0, 2*pi], angles outside these intervals are supported and handled via
    periodic extension by SciPy's sph_harm_y (periodic reflection across poles for theta,
    and modulo 2*pi continuation for phi). Both theta and phi must contain finite numbers.

    Raises
    ------
    ValueError
        If quantum numbers are invalid or angles contain non-finite values (NaN, Inf).
    """
    if not (isinstance(l, (int, np.integer)) and not isinstance(l, bool) and l >= 0):
        raise ValueError(f"Orbital quantum number l must be an integer >= 0 (got {l}).")
    if not (isinstance(m, (int, np.integer)) and not isinstance(m, bool) and -l <= m <= l):
        raise ValueError(f"Magnetic quantum number m must satisfy -{l} <= m <= {l} (got m={m}).")

    th_arr = np.asarray(theta, dtype=float)
    ph_arr = np.asarray(phi, dtype=float)

    if not (np.all(np.isfinite(th_arr)) and np.all(np.isfinite(ph_arr))):
        raise ValueError("Angles theta and phi must contain only finite numbers.")

    val = sph_harm_y(int(l), int(m), th_arr, ph_arr)

    if np.ndim(theta) == 0 and np.ndim(phi) == 0:
        return complex(val.item())
    return val


def real_angular_wavefunction(
    l: int,
    orbital: str,
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute real-valued spherical harmonics for standard named orbitals.

    Uses the standard Condon-Shortley phase convention matching SciPy's sph_harm_y:
        Y_l^{-m} = (-1)^m (Y_l^m)*

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number (0, 1, 2, or 3).
    orbital : str
        Named orbital identifier:
        - l=0: "s"
        - l=1: "p_x", "p_y", "p_z"
        - l=2: "d_z2", "d_xz", "d_yz", "d_xy", "d_x2-y2"
        - l=3: "f_z3", "f_xz2", "f_yz2", "f_xyz", "f_z(x2-y2)", "f_x(x2-3y2)", "f_y(3x2-y2)"
    theta : float or numpy.ndarray
        Polar (colatitudinal) angle in radians.
    phi : float or numpy.ndarray
        Azimuthal angle in radians.

    Returns
    -------
    float or numpy.ndarray
        Real-valued spherical harmonic value(s). Returns a float if both angles
        are scalar, or an ndarray if either is an array.

    Notes
    -----
    Phase Convention and Linear Combinations:
    Under the Condon-Shortley convention, Y_l^m includes a (-1)^m phase factor for m > 0.
    The exact linear combinations of complex spherical harmonics Y_l^m used are:

    l = 0:
      - "s"            = Y_0^0

    l = 1 (p orbitals):
      - "p_z"          = Y_1^0
      - "p_x"          = (1 / sqrt(2)) * (Y_1^{-1} - Y_1^1)
      - "p_y"          = (i / sqrt(2)) * (Y_1^{-1} + Y_1^1)

    l = 2 (d orbitals):
      - "d_z2"         = Y_2^0
      - "d_xz"         = (1 / sqrt(2)) * (Y_2^{-1} - Y_2^1)
      - "d_yz"         = (i / sqrt(2)) * (Y_2^{-1} + Y_2^1)
      - "d_xy"         = (i / sqrt(2)) * (Y_2^{-2} - Y_2^2)
      - "d_x2-y2"      = (1 / sqrt(2)) * (Y_2^{-2} + Y_2^2)

    l = 3 (f orbitals):
      - "f_z3"         = Y_3^0
      - "f_xz2"        = (1 / sqrt(2)) * (Y_3^{-1} - Y_3^1)
      - "f_yz2"        = (i / sqrt(2)) * (Y_3^{-1} + Y_3^1)
      - "f_xyz"        = (i / sqrt(2)) * (Y_3^{-2} - Y_3^2)
      - "f_z(x2-y2)"   = (1 / sqrt(2)) * (Y_3^{-2} + Y_3^2)
      - "f_x(x2-3y2)"  = (1 / sqrt(2)) * (Y_3^{-3} - Y_3^3)
      - "f_y(3x2-y2)"  = (i / sqrt(2)) * (Y_3^{-3} + Y_3^3)

    These combinations yield strictly real-valued spatial functions corresponding to
    the standard Cartesian lobe distributions visualized in chemistry and physics.

    Raises
    ------
    ValueError
        If l is not in {0, 1, 2, 3}, orbital name is unrecognized for the given l,
        or angles contain non-finite values (NaN, Inf).
    """
    valid_orbitals = {
        0: ["s"],
        1: ["p_x", "p_y", "p_z"],
        2: ["d_z2", "d_xz", "d_yz", "d_xy", "d_x2-y2"],
        3: ["f_z3", "f_xz2", "f_yz2", "f_xyz", "f_z(x2-y2)", "f_x(x2-3y2)", "f_y(3x2-y2)"],
    }

    if not (isinstance(l, (int, np.integer)) and not isinstance(l, bool) and l in valid_orbitals):
        raise ValueError(f"Angular quantum number l must be an integer in {{0, 1, 2, 3}} (got {l}).")

    if not isinstance(orbital, str) or orbital not in valid_orbitals[l]:
        raise ValueError(
            f"Invalid orbital name '{orbital}' for l={l}. "
            f"Expected one of: {valid_orbitals[l]}"
        )

    th_arr = np.asarray(theta, dtype=float)
    ph_arr = np.asarray(phi, dtype=float)

    if not (np.all(np.isfinite(th_arr)) and np.all(np.isfinite(ph_arr))):
        raise ValueError("Angles theta and phi must contain only finite numbers.")

    s2 = np.sqrt(2.0)

    # Evaluate the linear combinations using sph_harm_y
    if l == 0:
        raw = sph_harm_y(0, 0, th_arr, ph_arr)
    elif l == 1:
        if orbital == "p_z":
            raw = sph_harm_y(1, 0, th_arr, ph_arr)
        elif orbital == "p_x":
            raw = (sph_harm_y(1, -1, th_arr, ph_arr) - sph_harm_y(1, 1, th_arr, ph_arr)) / s2
        elif orbital == "p_y":
            raw = 1j * (sph_harm_y(1, -1, th_arr, ph_arr) + sph_harm_y(1, 1, th_arr, ph_arr)) / s2
    elif l == 2:
        if orbital == "d_z2":
            raw = sph_harm_y(2, 0, th_arr, ph_arr)
        elif orbital == "d_xz":
            raw = (sph_harm_y(2, -1, th_arr, ph_arr) - sph_harm_y(2, 1, th_arr, ph_arr)) / s2
        elif orbital == "d_yz":
            raw = 1j * (sph_harm_y(2, -1, th_arr, ph_arr) + sph_harm_y(2, 1, th_arr, ph_arr)) / s2
        elif orbital == "d_xy":
            raw = 1j * (sph_harm_y(2, -2, th_arr, ph_arr) - sph_harm_y(2, 2, th_arr, ph_arr)) / s2
        elif orbital == "d_x2-y2":
            raw = (sph_harm_y(2, -2, th_arr, ph_arr) + sph_harm_y(2, 2, th_arr, ph_arr)) / s2
    elif l == 3:
        if orbital == "f_z3":
            raw = sph_harm_y(3, 0, th_arr, ph_arr)
        elif orbital == "f_xz2":
            raw = (sph_harm_y(3, -1, th_arr, ph_arr) - sph_harm_y(3, 1, th_arr, ph_arr)) / s2
        elif orbital == "f_yz2":
            raw = 1j * (sph_harm_y(3, -1, th_arr, ph_arr) + sph_harm_y(3, 1, th_arr, ph_arr)) / s2
        elif orbital == "f_xyz":
            raw = 1j * (sph_harm_y(3, -2, th_arr, ph_arr) - sph_harm_y(3, 2, th_arr, ph_arr)) / s2
        elif orbital == "f_z(x2-y2)":
            raw = (sph_harm_y(3, -2, th_arr, ph_arr) + sph_harm_y(3, 2, th_arr, ph_arr)) / s2
        elif orbital == "f_x(x2-3y2)":
            raw = (sph_harm_y(3, -3, th_arr, ph_arr) - sph_harm_y(3, 3, th_arr, ph_arr)) / s2
        elif orbital == "f_y(3x2-y2)":
            raw = 1j * (sph_harm_y(3, -3, th_arr, ph_arr) + sph_harm_y(3, 3, th_arr, ph_arr)) / s2

    # Verify that the imaginary component is zero within numerical floating-point precision
    if not np.allclose(np.imag(raw), 0.0, atol=1e-12):
        raise ValueError(f"Orbital combination for '{orbital}' produced a non-zero imaginary part.")

    real_val = np.real(raw)

    if np.ndim(theta) == 0 and np.ndim(phi) == 0:
        return float(real_val.item())
    return real_val


def psi_nlm(
    n: int,
    l: int,
    m: int,
    r: Union[float, np.ndarray],
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
) -> Union[complex, np.ndarray]:
    """Compute the total hydrogen wavefunction psi_nlm(r, theta, phi) = R_nl(r) * Y_l^m(theta, phi).

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Orbital quantum number (0 <= l < n).
    m : int
        Magnetic quantum number (-l <= m <= l).
    r : float or numpy.ndarray
        Radial distance from nucleus in atomic units (a0 = 1).
    theta : float or numpy.ndarray
        Polar angle in radians [0, pi].
    phi : float or numpy.ndarray
        Azimuthal angle in radians [0, 2*pi].

    Returns
    -------
    complex or numpy.ndarray
        Wavefunction value(s). Returns a scalar complex if all inputs are scalar,
        or an ndarray if any input is an array.
    """
    r_wave = radial_wavefunction(n, l, r)
    ang_wave = angular_wavefunction(l, m, theta, phi)
    out = r_wave * ang_wave
    if np.ndim(out) == 0:
        return complex(out)
    return out


def expectation_values(n: int, l: int) -> Dict[str, Any]:
    """Compute radial expectation values <r>, <r^2>, and <1/r> using adaptive quadrature (quad),
    and compare against exact analytical solutions in atomic units (a0 = 1).

    Analytical formulas:
      <r>   = (1 / 2) * [3*n^2 - l*(l + 1)]
      <r^2> = (n^2 / 2) * [5*n^2 + 1 - 3*l*(l + 1)]
      <1/r> = 1 / n^2

    Returns
    -------
    dict
        Dictionary containing numerical values, quadrature error estimates,
        and analytical targets for norm, <r>, <r^2>, and <1/r>.
    """
    if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n >= 1):
        raise ValueError(f"Principal quantum number n must be an integer >= 1 (got {n}).")
    if not (isinstance(l, (int, np.integer)) and not isinstance(l, bool) and 0 <= l < n):
        raise ValueError(f"Orbital quantum number l must satisfy 0 <= l < n (got l={l}, n={n}).")

    # The radial probability density is P(r) = r^2 * |R_nl(r)|^2
    def p_density(radius: float) -> float:
        val = radial_wavefunction(n, l, radius)
        return float((radius ** 2) * (val ** 2))

    # Compute radial normalization integral over [0, inf) with explicit error tolerances.
    # Note: points cannot be combined with infinite integration limits in QUADPACK;
    # since P(r) ~ r^(2l+2) is smooth at r=0 and decays exponentially as r -> inf,
    # standard adaptive Gauss-Kronrod quadrature handles [0, inf) to machine precision.
    norm_val, norm_err = scipy.integrate.quad(
        p_density, 0.0, np.inf, epsabs=1e-11, epsrel=1e-11
    )

    # <r> = integral(r * P(r) dr)
    r_val, r_err = scipy.integrate.quad(
        lambda r: r * p_density(r), 0.0, np.inf, epsabs=1e-11, epsrel=1e-11
    )

    # <r^2> = integral(r^2 * P(r) dr)
    r2_val, r2_err = scipy.integrate.quad(
        lambda r: (r ** 2) * p_density(r), 0.0, np.inf, epsabs=1e-11, epsrel=1e-11
    )

    # <1/r> = integral((1/r) * P(r) dr)
    # Identity: |psi|^2 / r * r^2 dr = r * |R_nl(r)|^2 dr
    # Evaluating r * |R_nl(r)|^2 directly avoids division by r and is numerically stable at r=0.
    def inv_r_integrand(radius: float) -> float:
        val = radial_wavefunction(n, l, radius)
        return float(radius * (val ** 2))

    inv_r_val, inv_r_err = scipy.integrate.quad(
        inv_r_integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-11
    )

    # Exact analytical benchmarks
    ana_r = 0.5 * (3.0 * (n ** 2) - l * (l + 1))
    ana_r2 = 0.5 * (n ** 2) * (5.0 * (n ** 2) + 1.0 - 3.0 * l * (l + 1))
    ana_inv_r = 1.0 / (n ** 2)

    return {
        "norm": {"value": norm_val, "error_estimate": norm_err},
        "<r>": {"numerical": r_val, "analytical": ana_r, "error_estimate": r_err},
        "<r^2>": {"numerical": r2_val, "analytical": ana_r2, "error_estimate": r2_err},
        "<1/r>": {"numerical": inv_r_val, "analytical": ana_inv_r, "error_estimate": inv_r_err},
    }


def separated_quad_normalization(n: int, l: int, m: int) -> Tuple[float, float, float, float]:
    """Compute total normalization using separated 1D radial and 2D angular quad integrals.

    Since the hydrogen wavefunction factorizes:
      integral |psi|^2 dV = [integral_0^inf r^2 |R_nl(r)|^2 dr] * [integral |Y_l^m|^2 sin(theta) dtheta dphi]

    Returns
    -------
    total_norm : float
        Product of radial and angular integrals (should equal 1.0).
    rad_norm : float
        Radial normalization integral value.
    ang_norm : float
        Angular normalization integral value.
    rad_err : float
        Quadrature error estimate for the radial integral.
    """
    # 1. Radial integral over [0, inf)
    def r_integrand(r: float) -> float:
        val = radial_wavefunction(n, l, r)
        return float((r ** 2) * (val ** 2))

    rad_norm, rad_err = scipy.integrate.quad(
        r_integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-11
    )

    # 2. Angular integral: integral_0^2pi dphi integral_0^pi dtheta sin(theta) |Y_l^m|^2
    # Because |Y_l^m(theta, phi)|^2 = |Y_l^m(theta, 0)|^2 is independent of phi,
    # the phi integral is exactly 2*pi.
    def theta_integrand(theta: float) -> float:
        val = angular_wavefunction(l, m, theta, 0.0)
        return float((abs(val) ** 2) * np.sin(theta))

    th_val, th_err = scipy.integrate.quad(
        theta_integrand, 0.0, np.pi, epsabs=1e-11, epsrel=1e-11
    )
    ang_norm = 2.0 * np.pi * th_val

    total_norm = rad_norm * ang_norm
    return total_norm, rad_norm, ang_norm, rad_err


def spherical_grid_integral(
    n: int,
    l: int,
    m: int,
    r_max: float = None,
    nr: int = 300,
    ntheta: int = 100,
    nphi: int = 100,
) -> float:
    """Compute the 3D volume integral of |psi_nlm|^2 over a spherical grid via the trapezoid rule.

    This finite-grid numerical integration is evaluated over [0, r_max], [0, pi], [0, 2*pi].
    Note: A finite cutoff is used here for grid sampling, unlike quad which integrates over [0, inf).

    Parameters
    ----------
    n, l, m : int
        Quantum numbers.
    r_max : float, optional
        Upper radial cutoff. If None, dynamically set to max(30.0, 4.0 * n^2).
    nr, ntheta, nphi : int
        Number of grid intervals along r, theta, and phi (must be >= 2).

    Returns
    -------
    float
        The numerically integrated probability over the grid.
    """
    if r_max is None:
        r_max = max(30.0, 4.0 * (n ** 2))

    if r_max <= 0.0:
        raise ValueError(f"Radial cutoff r_max must be positive (got {r_max}).")
    if nr < 2 or ntheta < 2 or nphi < 2:
        raise ValueError(f"Grid sizes must be at least 2 (got nr={nr}, ntheta={ntheta}, nphi={nphi}).")

    r_grid = np.linspace(0.0, r_max, nr)
    theta_grid = np.linspace(0.0, np.pi, ntheta)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, nphi)

    R, THETA, PHI = np.meshgrid(r_grid, theta_grid, phi_grid, indexing="ij")
    psi = psi_nlm(n, l, m, R, THETA, PHI)
    prob_density = np.abs(psi) ** 2

    # Volume element: dV = r^2 * sin(theta) * dr * dtheta * dphi
    integrand = prob_density * (R ** 2) * np.sin(THETA)

    int_phi = scipy.integrate.trapezoid(integrand, phi_grid, axis=2)
    int_theta = scipy.integrate.trapezoid(int_phi, theta_grid, axis=1)
    total_int = scipy.integrate.trapezoid(int_theta, r_grid, axis=0)

    return float(total_int)


def find_radial_nodes(n: int, l: int, r_max: float = None) -> List[float]:
    """Find exact radial node positions (roots of R_nl(r) = 0 for r > 0) using Brent's method.

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Orbital quantum number (0 <= l < n).
    r_max : float, optional
        Upper search radius. Defaults to max(25.0, 4.0 * n^2).

    Returns
    -------
    list of float
        Sorted list of exact radial node radii. Length equals the theoretical node count n - l - 1.

    Raises
    ------
    ValueError
        If quantum numbers are invalid or r_max is non-positive.
    """
    # Trigger existing rigorous quantum number validations
    radial_wavefunction(n, l, 0.0)

    if r_max is not None and r_max <= 0.0:
        raise ValueError(f"Radial search limit r_max must be positive (got {r_max}).")

    num_nodes = n - l - 1
    if num_nodes == 0:
        return []

    if r_max is None:
        r_max = max(25.0, 4.0 * (n ** 2))

    # Dense sampling search grid to detect sign-change brackets
    r_search = np.linspace(1e-4, r_max, 5000)
    R_vals = radial_wavefunction(n, l, r_search)

    sign_changes = np.where(np.diff(np.sign(R_vals)))[0]
    nodes: List[float] = []

    for idx in sign_changes:
        r_a = r_search[idx]
        r_b = r_search[idx + 1]
        root = scipy.optimize.brentq(lambda r: radial_wavefunction(n, l, r), r_a, r_b)
        nodes.append(float(root))

    return nodes


if __name__ == "__main__":
    print("=" * 90)
    print("HYDROGEN WAVEFUNCTION IMPLEMENTATION AND VALIDATION DEMO")
    print("=" * 90)

    demo_states = [
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (2, 1, 1),
        (3, 0, 0),
        (3, 2, -1),
        (3, 2, 2),
    ]

    header = (
        f"{'State (n,l,m)':<15} | {'Norm (Quad)':<12} | {'Norm Err':<10} | "
        f"{'<r> (Num / Ana)':<18} | {'<r^2> (Num / Ana)':<18} | {'Quad Err':<10}"
    )
    print(header)
    print("-" * 90)

    for n_val, l_val, m_val in demo_states:
        exp = expectation_values(n_val, l_val)
        norm_val = exp["norm"]["value"]
        norm_err = abs(norm_val - 1.0)

        num_r = exp["<r>"]["numerical"]
        ana_r = exp["<r>"]["analytical"]
        num_r2 = exp["<r^2>"]["numerical"]
        ana_r2 = exp["<r^2>"]["analytical"]
        quad_err = exp["<r>"]["error_estimate"]

        print(
            f"({n_val}, {l_val}, {m_val:>2})        | "
            f"{norm_val:<12.8f} | {norm_err:<10.2e} | "
            f"{num_r:.2f} / {ana_r:<8.2f} | "
            f"{num_r2:.1f} / {ana_r2:<8.1f} | "
            f"{quad_err:<10.2e}"
        )

    print("-" * 90)
    print("Validation run completed successfully.")
