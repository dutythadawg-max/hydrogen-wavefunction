"""Hydrogen Wavefunction Implementation and Validation: 3D Orbital Volume Visualizer.

Computes 3D probability density |psi_nlm(r, theta, phi)|^2 on a Cartesian grid
and performs volume rendering using PyVista.
"""

import argparse
import sys
import numpy as np
import pyvista as pv

from a0 import psi_nlm


def create_orbital_grid(n: int = 3, l: int = 2, m: int = 0, grid_size: int = 80, extent: float = 25.0):
    """Generate a PyVista ImageData volume grid of the orbital probability density.

    Parameters
    ----------
    n : int
        Principal quantum number.
    l : int
        Orbital quantum number.
    m : int
        Magnetic quantum number.
    grid_size : int
        Number of grid points along each axis (grid_size x grid_size x grid_size).
    extent : float
        Half-width of the spatial domain in atomic units [-extent, extent].

    Returns
    -------
    pv.ImageData
        PyVista ImageData grid containing the probability density scalar field.
    """
    print(f"Sampling hydrogen orbital state: (n, l, m) = ({n}, {l}, {m})")
    print(f"Grid resolution: {grid_size}x{grid_size}x{grid_size}, extent: [-{extent}, {extent}] a0")

    coords = np.linspace(-extent, extent, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Transform Cartesian coordinates to spherical coordinates
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R_safe = np.where(R == 0.0, 1e-12, R)
    THETA = np.arccos(np.clip(Z / R_safe, -1.0, 1.0))
    PHI = np.mod(np.arctan2(Y, X), 2.0 * np.pi)

    # Compute total wavefunction and probability density
    psi = psi_nlm(n, l, m, R, THETA, PHI)
    prob_density = np.abs(psi) ** 2

    # Construct PyVista ImageData grid
    spacing = (2.0 * extent) / (grid_size - 1)
    grid = pv.ImageData()
    grid.dimensions = (grid_size, grid_size, grid_size)
    grid.origin = (-extent, -extent, -extent)
    grid.spacing = (spacing, spacing, spacing)

    # Assign scalar point data in Fortran order (x fastest, then y, then z)
    grid.point_data["probability_density"] = prob_density.flatten(order="F")

    print(f"Computed probability density: max={prob_density.max():.6e}, min={prob_density.min():.6e}")
    return grid


def visualize_orbital(
    n: int = 3,
    l: int = 2,
    m: int = 0,
    grid_size: int = 80,
    extent: float = 25.0,
    screenshot_path: str = None,
    off_screen: bool = False,
):
    """Render the 3D orbital probability density as a volumetric cloud.

    Parameters
    ----------
    n, l, m : int
        Quantum numbers.
    grid_size : int
        Grid resolution per dimension.
    extent : float
        Half-width of coordinate box.
    screenshot_path : str, optional
        Path to save screenshot (e.g. 'orbital_3d_volume.png').
    off_screen : bool
        Whether to run offscreen without opening an interactive GUI window.
    """
    grid = create_orbital_grid(n=n, l=l, m=m, grid_size=grid_size, extent=extent)

    # Opacity transfer function: suppress low-density background noise
    # and smoothly ramp up opacity inside orbital lobes
    opacity = [0.0, 0.0, 0.01, 0.05, 0.15, 0.35, 0.65, 0.95]

    plotter = pv.Plotter(window_size=[1000, 800], off_screen=off_screen)
    plotter.set_background("#101014")

    plotter.add_volume(
        grid,
        scalars="probability_density",
        cmap="plasma",
        opacity=opacity,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": r"|ψ|²",
            "vertical": True,
            "title_font_size": 14,
            "label_font_size": 10,
            "color": "white",
        },
    )

    plotter.add_text(
        f"Hydrogen Wavefunction: (n, l, m) = ({n}, {l}, {m})\nProbability Density |ψ|²",
        position="upper_left",
        font_size=11,
        color="white",
    )
    plotter.add_axes(color="white")
    plotter.add_bounding_box(color="gray")

    plotter.camera_position = [(60.0, 60.0, 45.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    if screenshot_path:
        plotter.screenshot(screenshot_path)
        print(f"Saved volume rendering screenshot to: {screenshot_path}")

    if not off_screen:
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydrogen Wavefunction 3D Volume Visualizer")
    parser.add_argument("-n", type=int, default=3, help="Principal quantum number (default: 3)")
    parser.add_argument("-l", type=int, default=2, help="Orbital quantum number (default: 2)")
    parser.add_argument("-m", type=int, default=0, help="Magnetic quantum number (default: 0)")
    parser.add_argument("--grid-size", type=int, default=80, help="Grid resolution (default: 80)")
    parser.add_argument("--extent", type=float, default=25.0, help="Spatial extent [-L, L] a0 (default: 25.0)")
    parser.add_argument("--screenshot", type=str, default="orbital_3d_volume.png", help="Path to save screenshot")
    parser.add_argument("--off-screen", action="store_true", help="Render offscreen without showing window")

    args = parser.parse_args()

    visualize_orbital(
        n=args.n,
        l=args.l,
        m=args.m,
        grid_size=args.grid_size,
        extent=args.extent,
        screenshot_path=args.screenshot,
        off_screen=args.off_screen,
    )
