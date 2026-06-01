#!/usr/bin/env python
"""Generate a schematic figure explaining the 4-AuNP pair-geometry feature conventions.

Features:
  d_1       -- intra-pair distance (shorter pair)
  d_2       -- intra-pair distance (longer pair)
  d_cc      -- center-to-center distance between pair centroids
  theta_rot -- signed angle between pair axes (-90 to +90 deg)
  theta_cc  -- signed angle from pair-1 axis to centroid-centroid vector (-90 to +90 deg)

Usage:
    uv run python src/make_feature_schematic.py
    uv run python src/make_feature_schematic.py -o fig.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def _draw_arc_arrow(ax, center, radius, theta1_deg, theta2_deg, color,
                    lw=2.5, label=None, label_radius=None, fontsize=16,
                    label_ha="center", label_va="center"):
    """Draw an arc from theta1 to theta2 (degrees) with an arrowhead at theta2."""
    theta = np.linspace(np.radians(theta1_deg), np.radians(theta2_deg), 80)
    xs = center[0] + radius * np.cos(theta)
    ys = center[1] + radius * np.sin(theta)
    ax.plot(xs, ys, color=color, lw=lw, zorder=4)

    # Arrowhead at the end
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-4], ys[-4]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=15),
                zorder=4)

    if label:
        mid_angle = np.radians((theta1_deg + theta2_deg) / 2)
        r = label_radius or (radius + 0.5)
        ax.text(center[0] + r * np.cos(mid_angle),
                center[1] + r * np.sin(mid_angle),
                label, fontsize=fontsize, color=color,
                ha=label_ha, va=label_va, fontweight="bold")


def make_schematic(output_path: Path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # --- 4 AuNP positions ---
    # Pair 1 (shorter): nearly horizontal, d_1 ~ 3.54
    a1 = np.array([0.0, 0.0])
    b1 = np.array([3.5, 0.5])

    # Pair 2 (longer, but similar): d_2 ~ 3.91, angled ~50 deg
    # Placed close to pair 1 (a2 near b1)
    a2 = np.array([5.0, 2.0])
    b2 = np.array([7.5, 5.0])

    # Derived quantities
    v1 = b1 - a1
    v2 = b2 - a2
    c1 = (a1 + b1) / 2
    c2 = (a2 + b2) / 2
    sep = c2 - c1
    dir1 = v1 / np.linalg.norm(v1)
    dir2 = v2 / np.linalg.norm(v2)
    sep_dir = sep / np.linalg.norm(sep)

    # Angles (for arc drawing)
    ang_v1 = np.degrees(np.arctan2(dir1[1], dir1[0]))
    ang_v2 = np.degrees(np.arctan2(dir2[1], dir2[0]))
    ang_sep = np.degrees(np.arctan2(sep_dir[1], sep_dir[0]))

    # Colors
    pair1_c = "#2166AC"
    pair2_c = "#B2182B"
    dcc_c = "#555555"
    trot_c = "#7F3F98"
    tcc_c = "#1A9641"
    gold = "#D4A017"
    gold_edge = "#8B6914"

    # ========================
    # Pair lines
    # ========================
    ax.plot(*zip(a1, b1), color=pair1_c, lw=3.5, zorder=3, solid_capstyle="round")
    ax.plot(*zip(a2, b2), color=pair2_c, lw=3.5, zorder=3, solid_capstyle="round")

    # ========================
    # d_1 dimension line (offset below pair 1)
    # ========================
    n1 = np.array([-dir1[1], dir1[0]])  # normal to pair 1
    p1s = a1 - 0.7 * n1
    p1e = b1 - 0.7 * n1
    ax.annotate("", xy=p1e, xytext=p1s,
                arrowprops=dict(arrowstyle="<->", color=pair1_c, lw=1.5))
    mid1 = (p1s + p1e) / 2
    ax.text(mid1[0], mid1[1] - 0.5, r"$d_1$", fontsize=16, color=pair1_c,
            ha="center", va="top", fontweight="bold")

    # ========================
    # d_2 dimension line (offset right of pair 2)
    # ========================
    n2 = np.array([-dir2[1], dir2[0]])  # normal to pair 2
    p2s = a2 + 0.7 * n2
    p2e = b2 + 0.7 * n2
    ax.annotate("", xy=p2e, xytext=p2s,
                arrowprops=dict(arrowstyle="<->", color=pair2_c, lw=1.5))
    mid2 = (p2s + p2e) / 2
    ax.text(mid2[0] + 0.5, mid2[1], r"$d_2$", fontsize=16, color=pair2_c,
            ha="left", va="center", fontweight="bold")

    # ========================
    # Centroids
    # ========================
    ax.scatter(*c1, s=100, marker="+", c=pair1_c, linewidths=2.5, zorder=4)
    ax.scatter(*c2, s=100, marker="+", c=pair2_c, linewidths=2.5, zorder=4)

    # ========================
    # d_cc dashed line between centroids
    # ========================
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color=dcc_c, lw=2,
            linestyle="--", zorder=2)
    mid_cc = (c1 + c2) / 2
    ax.text(mid_cc[0] - 0.3, mid_cc[1] + 0.6, r"$d_{cc}$", fontsize=16,
            color=dcc_c, ha="center", va="bottom", fontweight="bold")

    # ========================
    # theta_cc: signed angle at c1 from pair-1 axis to sep vector
    # ========================
    ext_cc = 3.0
    # Extension of pair-1 axis through c1 (dotted)
    ax.plot([c1[0] - 0.5 * dir1[0], c1[0] + ext_cc * dir1[0]],
            [c1[1] - 0.5 * dir1[1], c1[1] + ext_cc * dir1[1]],
            color=pair1_c, lw=1.5, linestyle=":", alpha=0.5, zorder=1)

    # Arc from v1 direction to sep direction (positive = CCW)
    _draw_arc_arrow(ax, c1, 2.0, ang_v1, ang_sep, tcc_c,
                    label=r"$\theta_{cc}$", label_radius=2.7)

    # ========================
    # theta_rot: signed angle between pair axes (detached diagram)
    # ========================
    arc_o = np.array([5.5, 5.8])
    ext = 2.2

    # Extension lines parallel to each pair axis
    ax.plot([arc_o[0], arc_o[0] + ext * dir1[0]],
            [arc_o[1], arc_o[1] + ext * dir1[1]],
            color=pair1_c, lw=1.5, linestyle=":", alpha=0.7, zorder=1)
    ax.plot([arc_o[0], arc_o[0] + ext * dir2[0]],
            [arc_o[1], arc_o[1] + ext * dir2[1]],
            color=pair2_c, lw=1.5, linestyle=":", alpha=0.7, zorder=1)

    # Arc from v1 direction to v2 direction (positive = CCW)
    _draw_arc_arrow(ax, arc_o, 1.5, ang_v1, ang_v2, trot_c,
                    label=r"$\theta_{rot}$", label_radius=2.2)

    # Small labels on extension lines
    ax.text(arc_o[0] + (ext + 0.3) * dir1[0],
            arc_o[1] + (ext + 0.3) * dir1[1],
            "|| pair 1", fontsize=7, color=pair1_c, alpha=0.7,
            ha="left", va="center")
    ax.text(arc_o[0] + (ext + 0.3) * dir2[0],
            arc_o[1] + (ext + 0.3) * dir2[1],
            "|| pair 2", fontsize=7, color=pair2_c, alpha=0.7,
            ha="left", va="center")

    # ========================
    # AuNP markers (no labels, just gold circles)
    # ========================
    for pt in [a1, b1, a2, b2]:
        ax.scatter(*pt, s=350, c=gold, edgecolors=gold_edge,
                   linewidths=1.5, zorder=5)

    # Label pairs near the pair lines
    pair1_mid = (a1 + b1) / 2
    ax.text(pair1_mid[0], pair1_mid[1] + 0.6, "pair 1", fontsize=9,
            color=pair1_c, ha="center", va="bottom", fontstyle="italic")
    pair2_mid = (a2 + b2) / 2
    ax.text(pair2_mid[0] - 0.7, pair2_mid[1], "pair 2", fontsize=9,
            color=pair2_c, ha="right", va="center", fontstyle="italic")

    # ========================
    # Legend
    # ========================
    desc = (
        "$d_1$: intra-pair distance (shorter pair, $d_1 \\leq d_2$)\n"
        "$d_2$: intra-pair distance (longer pair)\n"
        "$d_{cc}$: center-to-center distance between pairs\n"
        "$\\theta_{rot}$: signed rotation between pair axes (\u221290\u00b0 to +90\u00b0)\n"
        "$\\theta_{cc}$: signed angle of pair-1 axis vs. c\u2013c vector (\u221290\u00b0 to +90\u00b0)"
    )
    ax.text(0.02, 0.02, desc, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85))

    ax.set_xlim(-2.5, 11)
    ax.set_ylim(-3, 8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved schematic to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate feature convention schematic")
    parser.add_argument("-o", "--output", default="output/feature_schematic.pdf",
                        help="Output file path (default: output/feature_schematic.pdf)")
    args = parser.parse_args()
    make_schematic(args.output)
