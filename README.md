Hydrogen Wavefunctions: Implementation and Validation

This repo computes non-relativistic hydrogen atom wavefunctions in atomic units, where the Bohr radius $a_0 = 1$. It builds the radial functions $R_{nl}(r)$ from associated Laguerre polynomials, the angular functions $Y_l^m(\theta, \phi)$ from spherical harmonics, and the full wavefunctions $\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)$. It also checks the results numerically against known analytical values. Please if you might fact check me since l am not a physicist and most of the formulas are from deepseek.thanks also dont hate pls 

Setup
Prerequisites
Python 3.10 or newer. The code has been tested on Python 3.14.
SciPy 1.15.0 or newer. This version is required for scipy.special.sph_harm_y, which replaced the older sph_harm and uses the standard $(\theta, \phi)$ convention.
Installation

From the repository directory, create a virtual environment and install the dependencies
Usage

Validation demo. Prints a table of states with normalization and expectation values:
#In bash
python a0.py
Tests.
 Runs the test suite:
#In bash
pytest tests/ -v

Plots. Produces radial probability densities $P(r) = r^2 |R_{nl}(r)|^2$, with nodes found by root-finding, and 3D surfaces of the angular density $|Y_l^m(\theta, \phi)|^2$:

#In bash
python plot_hydrogen.py

3D volumetric rendering. Renders the probability cloud $|\psi|^2$ with PyVista:

#both should be run in bash 
#Interactive viewer (defaults to (n, l, m) = (3, 2, 0)):
python visualize_orbital.py

#Headless screenshot:
python visualize_orbital.py -n 2 -l 1 -m 0 --off-screen --screenshot orbital_3d_volume.png

Validation Results

Normalization integrals and radial expectation values are computed with adaptive quadrature (scipy.integrate.quad over $[0, \infty)$) and compared to the exact expressions:$$\langle r \rangle = \frac{a_0}{2} [3n^2 - l(l+1)], \quad \langle r^2 \rangle = \frac{a_0^2 n^2}{2} [5n^2 + 1 - 3l(l+1)], \quad \langle 1/r \rangle = \frac{1}{n^2 a_0}$$



Real Orbital Shapes

Chemistry textbooks usually draw orbitals as real-valued lobes ($p_x$, $p_y$, $p_z$, $d_{z^2}$, $d_{xz}$, $d_{yz}$, $d_{xy}$, $d_{x^2-y^2}$, and so on). These come from combining the complex harmonics.

Good quantum numbers versus chemical intuition. The complex $Y_l^m$ are simultaneous eigenstates of $\hat{L}^2$ and $\hat{L}_z$. For $m \neq 0$, though, their probability densities are rotationally symmetric tori around the $z$-axis, which doesn’t look like the familiar directed lobes. Those lobes are real linear combinations:

$$p_x = \frac{1}{\sqrt{2}} \left( Y_1^{-1} - Y_1^1 \right), \quad p_y = \frac{i}{\sqrt{2}} \left( Y_1^{-1} + Y_1^1 \right), \quad d_{x^2-y^2} = \frac{1}{\sqrt{2}} \left( Y_2^{-2} + Y_2^2 \right)$$
$m = \pm 1$ is not $p_x$ or $p_y$. It’s a common shortcut to match $m = +1$ with $p_x$ and $m = -1$ with $p_y$, but that’s wrong. Each of $p_x$ and $p_y$ is an equal-weight mix of $m = +1$ and $m = -1$.
Real orbitals are not $\hat{L}_z$ eigenstates. Because they mix $+m$ and $-m$, $\langle \hat{L}_z \rangle = 0$ and $m$ is not well defined. They are still eigenstates of $\hat{L}^2$, with eigenvalue $\hbar^2 l(l+1)$.
Limitations
Accuracy near $r = 0$ on uniform grids. Since $R_{nl}(r) \propto r^l$ near the origin, the radial volume integrand $r^2 |R_{nl}(r)|^2$ goes like $r^{2l+2}$. Uniform Cartesian or spherical grids can produce truncation and discretization errors there, especially for large $l$. The high-precision checks handle radial and angular parts separately and use adaptive Gauss-Kronrod quadrature (scipy.integrate.quad) for the radial integrals.
Radial cutoff depends on $n$. Excited states spread out to roughly $r \sim n^2 a_0$ and have long exponential tails. For finite-domain integration or volume rendering, the cutoff needs to grow with $n$, for example $r_{\max} \ge \max(30, 4n^2) \, a_0$, or the density gets cut off artificially.
Use $|\psi|^2$ for $m \neq 0$. Spherical harmonics with $m \neq 0$ are complex, since they carry a $e^{i m \phi}$ phase. Probability densities must be computed as $|\psi|^2 = \psi^* \psi = \text{np.abs}(\psi)^2$. Squaring $\psi$ directly gives wrong results.
SciPy version. scipy.special.sph_harm_y only exists in SciPy 1.15.0 and later. Older versions will raise an ImportError.

