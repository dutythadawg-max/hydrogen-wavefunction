"""Test Suite for Hydrogen Wavefunction Implementation and Validation.

Tests are organized into dedicated test classes separating input validation,
type handling, normalization checks, expectation-value comparisons,
orthogonality, and numerical convergence.
"""

import numpy as np
import pytest
import scipy.integrate

from a0 import (
    radial_wavefunction,
    angular_wavefunction,
    real_angular_wavefunction,
    psi_nlm,
    expectation_values,
    separated_quad_normalization,
    spherical_grid_integral,
    find_radial_nodes,
)

MINIMAL_STATES = [
    (1, 0, 0),
    (2, 0, 0),
    (2, 1, 0),
    (2, 1, -1),
    (2, 1, 1),
    (3, 0, 0),
    (3, 2, -1),
    (3, 2, 1),
    (3, 2, -2),
    (3, 2, 2),
]


# =====================================================================
# 1. Input Validation Tests
# =====================================================================
class TestInputValidation:
    """Verify robust exception handling for invalid inputs with direct, unguarded assertions."""

    @pytest.mark.parametrize("n, l", [
        (0, 0),        # n < 1
        (-1, 0),       # n < 0
        (1.5, 0),      # non-integer n
        (True, 0),     # boolean n
        (2, 2),        # l >= n
        (2, 3),        # l > n
        (2, -1),       # l < 0
        (2, 1.0),      # non-integer l
        (2, False),    # boolean l
    ])
    def test_invalid_radial_quantum_numbers(self, n, l):
        """Verify that every invalid (n, l) pair raises ValueError without conditional guards."""
        with pytest.raises(ValueError):
            radial_wavefunction(n, l, 1.0)

    @pytest.mark.parametrize("l, m", [
        (-1, 0),       # l < 0
        (1.5, 0),      # non-integer l
        (True, 0),     # boolean l
        (1, 2),        # |m| > l
        (1, -2),       # |m| > l
        (2, 3),        # |m| > l
        (2, -3),       # |m| > l
        (2, 0.5),      # non-integer m
        (1, 0.5),      # non-integer m
        (2, -1.5),     # non-integer m
        (1, True),     # boolean m
    ])
    def test_invalid_angular_quantum_numbers(self, l, m):
        """Verify that every invalid (l, m) pair raises ValueError without conditional guards."""
        with pytest.raises(ValueError):
            angular_wavefunction(l, m, 0.5, 0.5)

    @pytest.mark.parametrize("invalid_r", [
        -1.0,
        [-0.1, 1.0],
        np.array([-2.0, 3.0]),
        np.nan,
        np.inf,
        -np.inf,
        [1.0, np.nan, 2.0],
        [1.0, np.inf, 2.0],
    ])
    def test_invalid_radial_inputs(self, invalid_r):
        with pytest.raises(ValueError):
            radial_wavefunction(1, 0, invalid_r)

    @pytest.mark.parametrize("invalid_angle", [
        np.nan,
        np.inf,
        [0.0, np.nan],
        [0.0, np.inf],
    ])
    def test_invalid_angular_inputs(self, invalid_angle):
        with pytest.raises(ValueError):
            angular_wavefunction(1, 0, invalid_angle, 0.0)
        with pytest.raises(ValueError):
            angular_wavefunction(1, 0, 0.0, invalid_angle)

    def test_invalid_grid_parameters(self):
        with pytest.raises(ValueError, match="cutoff"):
            spherical_grid_integral(1, 0, 0, r_max=0.0)
        with pytest.raises(ValueError, match="cutoff"):
            spherical_grid_integral(1, 0, 0, r_max=-5.0)
        with pytest.raises(ValueError, match="Grid size"):
            spherical_grid_integral(1, 0, 0, nr=1)
        with pytest.raises(ValueError, match="Grid size"):
            spherical_grid_integral(1, 0, 0, ntheta=1)

    @pytest.mark.parametrize("n, l, r_max", [
        (0, 0, 10.0),    # n < 1
        (2, 2, 10.0),    # l >= n
        (2, -1, 10.0),   # l < 0
        (2, 0, 0.0),     # r_max == 0
        (2, 0, -5.0),    # r_max < 0
    ])
    def test_find_radial_nodes_invalid_inputs(self, n, l, r_max):
        """Verify find_radial_nodes raises ValueError for invalid quantum numbers or non-positive r_max."""
        with pytest.raises(ValueError):
            find_radial_nodes(n, l, r_max=r_max)


# =====================================================================
# 2. Shape and Type Consistency Tests
# =====================================================================
class TestShapesAndTypes:
    """Verify that scalar vs. array inputs strictly return the expected shapes and types."""

    def test_radial_scalar_vs_array(self):
        # Python scalar float
        out_float = radial_wavefunction(1, 0, 2.5)
        assert isinstance(out_float, float)

        # 0-d numpy array
        out_0d = radial_wavefunction(1, 0, np.array(2.5))
        assert isinstance(out_0d, float)

        # Size-1 array: must return an ndarray of shape (1,), not a float
        out_size1 = radial_wavefunction(1, 0, np.array([2.5]))
        assert isinstance(out_size1, np.ndarray)
        assert out_size1.shape == (1,)

        # Multi-dimensional array
        arr = np.linspace(0.0, 5.0, 24).reshape((2, 3, 4))
        out_arr = radial_wavefunction(2, 1, arr)
        assert isinstance(out_arr, np.ndarray)
        assert out_arr.shape == arr.shape

    def test_angular_scalar_vs_array(self):
        # Scalar angles
        out_scalar = angular_wavefunction(1, 0, 0.5, 1.2)
        assert isinstance(out_scalar, complex)

        # Size-1 arrays
        out_size1 = angular_wavefunction(1, 0, np.array([0.5]), np.array([1.2]))
        assert isinstance(out_size1, np.ndarray)
        assert out_size1.shape == (1,)

    def test_psi_scalar_vs_array(self):
        # All scalars
        out_scalar = psi_nlm(1, 0, 0, 1.0, 0.5, 0.5)
        assert isinstance(out_scalar, complex)

        # Size-1 array input
        out_size1 = psi_nlm(1, 0, 0, np.array([1.0]), 0.5, 0.5)
        assert isinstance(out_size1, np.ndarray)
        assert out_size1.shape == (1,)

    def test_angular_periodic_extension(self):
        """Verify that out-of-range angles are handled via periodic continuation."""
        theta = 0.8
        phi = 1.3
        val_base = angular_wavefunction(1, 1, theta, phi)
        val_shifted_phi = angular_wavefunction(1, 1, theta, phi + 2.0 * np.pi)
        assert np.isclose(val_base, val_shifted_phi, atol=1e-12)


# =====================================================================
# 3. Wavefunction Normalization Tests (Separate from Expectation Values)
# =====================================================================
class TestWavefunctionNormalization:
    """Verify that total, radial, and angular wavefunctions integrate to 1."""

    @pytest.mark.parametrize("n, l, m", MINIMAL_STATES)
    def test_separated_quad_normalization(self, n, l, m):
        total_norm, rad_norm, ang_norm, rad_err = separated_quad_normalization(n, l, m)

        assert np.isclose(rad_norm, 1.0, atol=1e-8), f"Radial norm failed for n={n}, l={l}: {rad_norm}"
        assert np.isclose(ang_norm, 1.0, atol=1e-8), f"Angular norm failed for l={l}, m={m}: {ang_norm}"
        assert np.isclose(total_norm, 1.0, atol=1e-8), f"Total norm failed for ({n},{l},{m}): {total_norm}"

    @pytest.mark.parametrize("l, m", [
        (0, 0),
        (1, 0),
        (1, 1),
        (1, -1),
        (2, 0),
        (2, 1),
        (2, -1),
        (2, 2),
        (2, -2),
        (3, 2),
    ])
    def test_angular_normalization_direct(self, l, m):
        """Direct 2D integration of |Y_l^m|^2 sin(theta) dtheta dphi over [0,pi]x[0,2pi]."""
        def integrand(theta, phi):
            val = angular_wavefunction(l, m, theta, phi)
            return float((abs(val) ** 2) * np.sin(theta))

        res, _ = scipy.integrate.dblquad(
            integrand,
            0.0,
            2.0 * np.pi,
            0.0,
            np.pi,
            epsabs=1e-9,
            epsrel=1e-9,
        )
        assert np.isclose(res, 1.0, atol=1e-7), f"Direct angular norm failed for ({l},{m}): {res}"

    @pytest.mark.parametrize("n, l, m", [
        (1, 0, 0),
        (2, 1, 0),
        (2, 1, 1),
        (3, 2, -1),
    ])
    def test_3d_spherical_grid_normalization(self, n, l, m):
        """Verify 3D grid trapezoidal integration approximates 1 with tightened atol=1e-4."""
        r_cutoff = max(30.0, 4.0 * (n ** 2))
        grid_norm = spherical_grid_integral(n, l, m, r_max=r_cutoff, nr=400, ntheta=200, nphi=100)
        assert np.isclose(grid_norm, 1.0, atol=1e-4), (
            f"Grid 3D norm failed for ({n},{l},{m}): {grid_norm} (diff: {abs(grid_norm - 1.0):.2e})"
        )

    def test_grid_convergence_monotonicity(self):
        """Assert normalization error on a fixed domain decreases monotonically across resolution levels."""
        n, l, m = 1, 0, 0
        r_max = 20.0
        # Three resolution levels well above float64 noise floor: [40, 80, 160]
        resolutions = [40, 80, 160]
        errors = []

        for nr in resolutions:
            val = spherical_grid_integral(n, l, m, r_max=r_max, nr=nr, ntheta=nr // 2, nphi=nr // 2)
            errors.append(abs(val - 1.0))

        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"Error did not decrease from nr={resolutions[i]} (err={errors[i]}) "
                f"to nr={resolutions[i+1]} (err={errors[i+1]})"
            )


# =====================================================================
# 4. Orbital Orthogonality Tests
# =====================================================================
class TestOrbitalOrthogonality:
    """Verify that distinct hydrogen wavefunctions are mutually orthogonal."""

    @pytest.mark.parametrize("state_a, state_b, expected_overlap", [
        ((1, 0, 0), (2, 0, 0), 0.0),   # Different n, same l, m
        ((2, 0, 0), (2, 1, 0), 0.0),   # Same n, different l
        ((2, 1, 0), (2, 1, 1), 0.0),   # Same n, l, different m
        ((2, 1, -1), (2, 1, 1), 0.0),  # Same n, l, different m
        ((1, 0, 0), (3, 2, 1), 0.0),   # Different n, l, m
        ((2, 1, 1), (2, 1, 1), 1.0),   # Identical states
        ((3, 2, -1), (3, 2, -1), 1.0), # Identical states
    ])
    def test_orthogonality(self, state_a, state_b, expected_overlap):
        n1, l1, m1 = state_a
        n2, l2, m2 = state_b

        # 1. Radial factor overlap
        def r_overlap(r):
            r1 = radial_wavefunction(n1, l1, r)
            r2 = radial_wavefunction(n2, l2, r)
            return float((r ** 2) * r1 * r2)

        rad_integral, _ = scipy.integrate.quad(r_overlap, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10)

        # 2. Angular factor overlap
        if m1 != m2:
            ang_integral = 0.0
        else:
            def th_overlap(theta):
                # Using phi=0 is safe because Y_l^m factors as P_l^m(cos theta)*e^(i*m*phi); at phi=0 the value is real-valued, making np.real exact.
                y1 = angular_wavefunction(l1, m1, theta, 0.0)
                y2 = angular_wavefunction(l2, m2, theta, 0.0)
                return float(np.real(np.conj(y1) * y2) * np.sin(theta))

            th_val, _ = scipy.integrate.quad(th_overlap, 0.0, np.pi, epsabs=1e-10, epsrel=1e-10)
            ang_integral = 2.0 * np.pi * th_val

        overlap = rad_integral * ang_integral
        assert np.isclose(overlap, expected_overlap, atol=1e-7), (
            f"Overlap for {state_a} and {state_b} expected {expected_overlap}, got {overlap}"
        )


# =====================================================================
# 5. Expectation Value Tests (Assumes normalization is verified)
# =====================================================================
class TestExpectationValues:
    """Verify radial expectation values <r>, <r^2>, and <1/r> against analytical results."""

    @pytest.mark.parametrize("n, l, m", MINIMAL_STATES)
    def test_expectation_values_against_analytic(self, n, l, m):
        exp = expectation_values(n, l)

        # 1. <r>
        num_r = exp["<r>"]["numerical"]
        ana_r = exp["<r>"]["analytical"]
        err_r = exp["<r>"]["error_estimate"]
        expected_ana_r = 0.5 * (3.0 * (n ** 2) - l * (l + 1))
        assert np.isclose(ana_r, expected_ana_r), f"Analytic formula mismatch for <r>: {ana_r} vs {expected_ana_r}"
        assert np.isclose(num_r, ana_r, atol=1e-6, rtol=1e-5), f"<r> failed for ({n},{l}): {num_r} vs {ana_r}"

        # 2. <r^2>
        num_r2 = exp["<r^2>"]["numerical"]
        ana_r2 = exp["<r^2>"]["analytical"]
        err_r2 = exp["<r^2>"]["error_estimate"]
        expected_ana_r2 = 0.5 * (n ** 2) * (5.0 * (n ** 2) + 1.0 - 3.0 * l * (l + 1))
        assert np.isclose(ana_r2, expected_ana_r2), f"Analytic formula mismatch for <r^2>: {ana_r2} vs {expected_ana_r2}"
        assert np.isclose(num_r2, ana_r2, atol=1e-5, rtol=1e-5), f"<r^2> failed for ({n},{l}): {num_r2} vs {ana_r2}"

        # 3. <1/r>
        num_inv_r = exp["<1/r>"]["numerical"]
        ana_inv_r = exp["<1/r>"]["analytical"]
        err_inv_r = exp["<1/r>"]["error_estimate"]
        expected_ana_inv_r = 1.0 / (n ** 2)
        assert np.isclose(ana_inv_r, expected_ana_inv_r), f"Analytic formula mismatch for <1/r>: {ana_inv_r} vs {expected_ana_inv_r}"
        assert np.isclose(num_inv_r, ana_inv_r, atol=1e-6, rtol=1e-5), f"<1/r> failed for ({n},{l}): {num_inv_r} vs {ana_inv_r}"

        # Quadrature error estimates threshold (raised to 1e-7 to prevent spurious failures)
        assert err_r < 1e-7, f"Quadrature error for <r> too large: {err_r}"
        assert err_r2 < 1e-7, f"Quadrature error for <r^2> too large: {err_r2}"
        assert err_inv_r < 1e-7, f"Quadrature error for <1/r> too large: {err_inv_r}"


# =====================================================================
# 6. Radial Node Location Tests
# =====================================================================
class TestRadialNodes:
    """Verify that root-finding accurately computes radial node positions and counts."""

    @pytest.mark.parametrize("n, l, expected_node_count", [
        (1, 0, 0),  # 1s
        (2, 0, 1),  # 2s
        (2, 1, 0),  # 2p
        (3, 0, 2),  # 3s
        (3, 1, 1),  # 3p
        (3, 2, 0),  # 3d
    ])
    def test_node_counts(self, n, l, expected_node_count):
        nodes = find_radial_nodes(n, l)
        assert len(nodes) == expected_node_count

    def test_exact_node_positions_2s_3p_3s(self):
        """Verify exact analytic node locations computed by root-finding.

        - 2s: R_20(r) = (1 / (2*sqrt(2))) * (2 - r) * exp(-r/2)
              => Exact root at r = 2.0 a0.
        - 3p: R_31(r) = (4 / (81*sqrt(6))) * (6*r - r^2) * exp(-r/3)
              => Exact root at r = 6.0 a0.
        - 3s: R_30(r) = (2 / (81*sqrt(3))) * (27 - 18*r + 2*r^2) * exp(-r/3)
              Roots of 2*r^2 - 18*r + 27 = 0 are r = (9 ± 3*sqrt(3)) / 2 = (3/2)*(3 ± sqrt(3)) a0.
              (which corresponds to rho = 3 ± sqrt(3) where rho = 2*r/3).
        """
        # 2s node
        nodes_2s = find_radial_nodes(2, 0)
        assert len(nodes_2s) == 1
        assert np.isclose(nodes_2s[0], 2.0, atol=1e-7)

        # 3p node
        nodes_3p = find_radial_nodes(3, 1)
        assert len(nodes_3p) == 1
        assert np.isclose(nodes_3p[0], 6.0, atol=1e-7)

        # 3s nodes
        nodes_3s = find_radial_nodes(3, 0)
        assert len(nodes_3s) == 2
        expected_3s_inner = 1.5 * (3.0 - np.sqrt(3.0))  # ~ 1.9019238 a0
        expected_3s_outer = 1.5 * (3.0 + np.sqrt(3.0))  # ~ 7.0980762 a0
        assert np.isclose(nodes_3s[0], expected_3s_inner, atol=1e-7)
        assert np.isclose(nodes_3s[1], expected_3s_outer, atol=1e-7)


# =====================================================================
# 7. Real-Valued Angular Wavefunction Tests
# =====================================================================
class TestRealAngularWavefunction:
    """Verify normalization, orthogonality, known values, and roundtrip of real spherical harmonics."""

    ORBITALS = [
        (0, "s"),
        (1, "p_x"),
        (1, "p_y"),
        (1, "p_z"),
        (2, "d_z2"),
        (2, "d_xz"),
        (2, "d_yz"),
        (2, "d_xy"),
        (2, "d_x2-y2"),
        (3, "f_z3"),
        (3, "f_xz2"),
        (3, "f_yz2"),
        (3, "f_xyz"),
        (3, "f_z(x2-y2)"),
        (3, "f_x(x2-3y2)"),
        (3, "f_y(3x2-y2)"),
    ]

    @pytest.mark.parametrize("l, orbital", ORBITALS)
    def test_real_orbital_normalization(self, l, orbital):
        """Verify that integral |real_Y|^2 sin(theta) dtheta dphi = 1 for all real orbitals."""
        def integrand(theta, phi):
            val = real_angular_wavefunction(l, orbital, theta, phi)
            return float((val ** 2) * np.sin(theta))

        norm, _ = scipy.integrate.dblquad(
            integrand, 0.0, 2.0 * np.pi, 0.0, np.pi, epsabs=1e-9, epsrel=1e-9
        )
        assert np.isclose(norm, 1.0, atol=1e-7), f"Real orbital {orbital} normalization failed: {norm}"

    @pytest.mark.parametrize("l, orb1, orb2", [
        (1, "p_x", "p_y"),
        (1, "p_x", "p_z"),
        (1, "p_y", "p_z"),
        (2, "d_z2", "d_xz"),
        (2, "d_z2", "d_xy"),
        (2, "d_xz", "d_yz"),
        (2, "d_xy", "d_x2-y2"),
        (3, "f_z3", "f_xyz"),
        (3, "f_xz2", "f_yz2"),
        (3, "f_x(x2-3y2)", "f_y(3x2-y2)"),
    ])
    def test_real_orbital_orthogonality(self, l, orb1, orb2):
        """Verify orthogonality: <orb1|orb2> = 0 for distinct real orbitals of the same l."""
        def integrand(theta, phi):
            y1 = real_angular_wavefunction(l, orb1, theta, phi)
            y2 = real_angular_wavefunction(l, orb2, theta, phi)
            return float(y1 * y2 * np.sin(theta))

        overlap, _ = scipy.integrate.dblquad(
            integrand, 0.0, 2.0 * np.pi, 0.0, np.pi, epsabs=1e-9, epsrel=1e-9
        )
        assert np.isclose(overlap, 0.0, atol=1e-8), f"Overlap between {orb1} and {orb2} non-zero: {overlap}"

    def test_known_values(self):
        """Verify exact known values at specific angles."""
        expected_val = np.sqrt(3.0 / (4.0 * np.pi))

        # p_z(theta=0, phi=0) = sqrt(3 / 4pi)
        pz_val = real_angular_wavefunction(1, "p_z", 0.0, 0.0)
        assert np.isclose(pz_val, expected_val, atol=1e-12)

        # p_x(theta=pi/2, phi=0) = sqrt(3 / 4pi)
        px_val_x = real_angular_wavefunction(1, "p_x", np.pi / 2.0, 0.0)
        assert np.isclose(px_val_x, expected_val, atol=1e-12)

        # p_x(theta=pi/2, phi=pi/2) = 0
        px_val_y = real_angular_wavefunction(1, "p_x", np.pi / 2.0, np.pi / 2.0)
        assert np.isclose(px_val_y, 0.0, atol=1e-12)

    def test_roundtrip_pz_equals_y10(self):
        """Verify roundtrip: p_z(theta, phi) == Y_1^0(theta, phi) identically."""
        theta_grid = np.linspace(0.0, np.pi, 10)
        phi_grid = np.linspace(0.0, 2.0 * np.pi, 10)
        TH, PH = np.meshgrid(theta_grid, phi_grid)

        pz_arr = real_angular_wavefunction(1, "p_z", TH, PH)
        y10_arr = angular_wavefunction(1, 0, TH, PH)

        assert np.allclose(np.imag(y10_arr), 0.0, atol=1e-12)
        assert np.allclose(pz_arr, np.real(y10_arr), atol=1e-12)

    @pytest.mark.parametrize("invalid_l, invalid_orbital", [
        (4, "g_z4"),
        (-1, "p_x"),
        (1, "d_xy"),
        (2, "p_z"),
        (3, "d_x2-y2"),
        (1, "invalid_name"),
    ])
    def test_invalid_orbital_inputs(self, invalid_l, invalid_orbital):
        with pytest.raises(ValueError):
            real_angular_wavefunction(invalid_l, invalid_orbital, 0.5, 0.5)

    def test_scalar_vs_array_real_orbitals(self):
        # Scalar input
        out_scalar = real_angular_wavefunction(1, "p_x", 0.5, 0.5)
        assert isinstance(out_scalar, float)

        # Size-1 array input
        out_size1 = real_angular_wavefunction(1, "p_x", np.array([0.5]), np.array([0.5]))
        assert isinstance(out_size1, np.ndarray)
        assert out_size1.shape == (1,)
        assert issubclass(out_size1.dtype.type, np.floating)
