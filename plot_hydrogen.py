"""Hydrogen Wavefunction Implementation and Validation: Plotting Module.

Generates:
1. radial_probability_density.png:
   Radial probability density P(r) = r^2 |R_nl(r)|^2 in atomic units (a0 = 1).
   Exact radial node locations computed via Brent's method root-finding (scipy.optimize.brentq).
2. angular_probability_density.png:
   3D angular probability density surfaces where the surface radius represents |Y_l^m(theta, phi)|^2.
   Note: Angular surfaces are dimensionless (unitless); surface radius represents angular probability
   density, not physical distance from the nucleus.
"""

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive rendering backend
import matplotlib.pyplot as plt
import numpy as np

from a0 import radial_wavefunction, angular_wavefunction, real_angular_wavefunction, find_radial_nodes


def plot_radial_densities():
    """Plot radial probability density P(r) = r^2 |R_nl(r)|^2 for 1s, 2s, 2p, 3d states."""
    states = [
        (1, 0, "(n, l) = (1, 0)", "tab:blue"),
        (2, 0, "(n, l) = (2, 0)", "tab:orange"),
        (2, 1, "(n, l) = (2, 1)", "tab:green"),
        (3, 2, "(n, l) = (3, 2)", "tab:red"),
    ]

    r_max = 25.0
    r = np.linspace(1e-4, r_max, 3000)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    print("\n--- Radial Probability Density & Exact Node Analysis ---")
    for n, l, label, color in states:
        expected_nodes = n - l - 1
        R = radial_wavefunction(n, l, r)
        P = (r ** 2) * (R ** 2)

        # Plot density curve
        ax.plot(r, P, label=f"{label} [analytic nodes: {expected_nodes}]", color=color, lw=2.0)

        # Compute exact radial nodes via root-finding (brentq)
        exact_nodes = find_radial_nodes(n, l, r_max=r_max)
        if exact_nodes:
            ax.scatter(
                exact_nodes,
                np.zeros_like(exact_nodes),
                color=color,
                s=60,
                marker="o",
                facecolors="none",
                lw=2,
                zorder=6,
            )

        # Find local peaks of P(r)
        is_peak = (P[1:-1] > P[:-2]) & (P[1:-1] > P[2:])
        peak_idx = np.where(is_peak)[0] + 1
        peak_r = r[peak_idx]
        peak_P = P[peak_idx]
        ax.scatter(peak_r, peak_P, color=color, s=40, zorder=5, marker="^")

        peaks_str = ", ".join([f"{p:.2f} a0" for p in peak_r])
        nodes_str = ", ".join([f"{nd:.4f} a0" for nd in exact_nodes]) if exact_nodes else "None"
        print(f"State {label:<17} | Analytic Nodes: {expected_nodes} | Exact Roots: {len(exact_nodes)} ({nodes_str}) | Maxima: {peaks_str}")

    ax.set_title(
        r"Radial Probability Density $P(r) = r^2 |R_{nl}(r)|^2$ (Hydrogen Wavefunction Validation)",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel(r"Radial Distance $r$ [atomic units, $a_0 = 1$]", fontsize=11)
    ax.set_ylabel(r"Radial Probability Density $P(r)$", fontsize=11)
    ax.set_xlim(0, r_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Legend annotations
    ax.plot([], [], "^", color="gray", label="Radial Peaks (Maxima)")
    ax.plot([], [], "o", markerfacecolor="none", markeredgecolor="gray", markeredgewidth=2, label="Exact Roots ($R_{nl}=0$, brentq)")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    output_path = "radial_probability_density.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved radial probability density plot to: {output_path}")


def plot_angular_densities():
    """Plot 3D angular probability density |Y_l^m(theta, phi)|^2 surfaces for representative states."""
    # Label panels strictly with quantum numbers (l, m); avoid real-orbital notation
    angular_states = [
        (1, 0, "(l, m) = (1, 0)"),
        (1, 1, "(l, m) = (1, ±1)"),
        (2, 0, "(l, m) = (2, 0)"),
        (2, 1, "(l, m) = (2, ±1)"),
        (2, 2, "(l, m) = (2, ±2)"),
    ]

    theta = np.linspace(0.0, np.pi, 120)
    phi = np.linspace(0.0, 2.0 * np.pi, 240)
    THETA, PHI = np.meshgrid(theta, phi)

    fig = plt.figure(figsize=(18, 4.5), dpi=150)

    print("\n--- Angular Probability Density 3D Surface Plots ---")
    for idx, (l, m, title) in enumerate(angular_states, 1):
        ax = fig.add_subplot(1, len(angular_states), idx, projection="3d")

        Y = angular_wavefunction(l, m, THETA, PHI)
        prob = np.abs(Y) ** 2

        # Note: Surface radius represents unitless angular probability density |Y_l^m|^2,
        # NOT physical distance from the nucleus.
        radius = prob
        X = radius * np.sin(THETA) * np.cos(PHI)
        Y_coord = radius * np.sin(THETA) * np.sin(PHI)
        Z = radius * np.cos(THETA)

        # Normalize colors per panel
        max_val = radius.max()
        norm_prob = radius / (max_val if max_val > 1e-12 else 1.0)
        colors = plt.cm.plasma(norm_prob)

        ax.plot_surface(
            X,
            Y_coord,
            Z,
            facecolors=colors,
            rstride=2,
            cstride=2,
            shade=True,
            alpha=0.9,
            antialiased=True,
        )

        # Set 1:1:1 aspect ratio so spherical harmonics are not geometrically distorted
        ax.set_box_aspect((1, 1, 1))

        max_limit = np.max([np.abs(X).max(), np.abs(Y_coord).max(), np.abs(Z).max()]) * 1.1
        if max_limit < 1e-6:
            max_limit = 0.1
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
        ax.set_zlim(-max_limit, max_limit)
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_axis_off()

    fig.suptitle(
        r"Angular Probability Density $|Y_l^m(\theta, \phi)|^2$ (Hydrogen Wavefunction Validation)",
        fontsize=13,
        y=0.98,
    )

    # Footnote explicitly clarifying units, surface radius meaning, and independent color/axis scaling
    fig.text(
        0.5,
        0.02,
        "Note: Surface radius represents unitless angular probability density |Y_l^m(θ, φ)|², not physical distance from the nucleus.\n"
        "Color scale and axis limits are scaled independently for each panel and are not directly comparable across different (l, m) states.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#333333",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    output_path = "angular_probability_density.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved angular probability density plot to: {output_path}")


def plot_real_orbitals():
    """Plot 3D angular probability density surfaces for real-valued orbitals (1s, 2p, 3d).

    Generates real_orbitals_1s2p3d.png showing the intuitive spatial lobe shapes
    (s, p_x, p_y, p_z, d_z2, d_xz, d_yz, d_xy, d_x2-y2) formed by superpositions of Y_l^m.
    """
    orbitals = [
        # (l, orbital_id, display_title, subplot_index)
        (0, "s", r"$s$ ($l=0$)", 1),
        (1, "p_x", r"$p_x$ ($l=1$)", 2),
        (1, "p_y", r"$p_y$ ($l=1$)", 3),
        (1, "p_z", r"$p_z$ ($l=1$)", 4),
        (2, "d_z2", r"$d_{z^2}$ ($l=2$)", 6),
        (2, "d_xz", r"$d_{xz}$ ($l=2$)", 7),
        (2, "d_yz", r"$d_{yz}$ ($l=2$)", 8),
        (2, "d_xy", r"$d_{xy}$ ($l=2$)", 9),
        (2, "d_x2-y2", r"$d_{x^2-y^2}$ ($l=2$)", 10),
    ]

    theta = np.linspace(0.0, np.pi, 120)
    phi = np.linspace(0.0, 2.0 * np.pi, 240)
    THETA, PHI = np.meshgrid(theta, phi)

    fig = plt.figure(figsize=(18, 8), dpi=150)

    print("\n--- Real Angular Orbitals 3D Surface Plots (1s, 2p, 3d) ---")
    for l, name, title, subplot_idx in orbitals:
        ax = fig.add_subplot(2, 5, subplot_idx, projection="3d")

        Y_real = real_angular_wavefunction(l, name, THETA, PHI)
        prob = Y_real ** 2

        # Surface radius represents unitless angular probability density |Y_real|^2
        radius = prob
        X = radius * np.sin(THETA) * np.cos(PHI)
        Y_coord = radius * np.sin(THETA) * np.sin(PHI)
        Z = radius * np.cos(THETA)

        # Normalize colors per panel
        max_val = radius.max()
        norm_prob = radius / (max_val if max_val > 1e-12 else 1.0)
        colors = plt.cm.viridis(norm_prob)

        ax.plot_surface(
            X,
            Y_coord,
            Z,
            facecolors=colors,
            rstride=2,
            cstride=2,
            shade=True,
            alpha=0.9,
            antialiased=True,
        )

        ax.set_box_aspect((1, 1, 1))

        max_limit = np.max([np.abs(X).max(), np.abs(Y_coord).max(), np.abs(Z).max()]) * 1.1
        if max_limit < 1e-6:
            max_limit = 0.1
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
        ax.set_zlim(-max_limit, max_limit)
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_axis_off()

    fig.suptitle(
        r"Real-Valued Angular Orbitals $|Y_{\mathrm{real}}(\theta, \phi)|^2$ (1s, 2p, 3d)",
        fontsize=14,
        y=0.98,
    )

    # Footnote explicitly clarifying units, surface radius meaning, and independent color/axis scaling
    fig.text(
        0.5,
        0.02,
        "Note: Surface radius represents unitless angular probability density |Y_real(θ, φ)|², not physical distance from the nucleus.\n"
        "Color scale and axis limits are scaled independently for each panel and are not directly comparable across different orbitals.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#333333",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path = "real_orbitals_1s2p3d.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved real orbital visualization plot to: {output_path}")


if __name__ == "__main__":
    plot_radial_densities()
    plot_angular_densities()
    plot_real_orbitals()
