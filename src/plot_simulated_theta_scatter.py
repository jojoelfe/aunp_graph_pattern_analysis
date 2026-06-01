#!/usr/bin/env python3
"""
Theta-scatter plot (theta_rot vs theta_cc) for simulated crystal and liquid AuNP patterns.

Loads particle positions from the solid_vs_liquid_2d simulations, applies Gaussian
noise (to mimic localisation uncertainty), extracts connected 4-node subgraphs, and
produces a two-panel hexbin density plot identical in style to plot_theta_scatter.py.

Usage:
    uv run python src/plot_simulated_theta_scatter.py
    uv run python src/plot_simulated_theta_scatter.py -o figures/sim_theta_scatter.png
    uv run python src/plot_simulated_theta_scatter.py --noise-sigma 1.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import starfile
import torch

from alignment import compute_pairwise_features
from graph_construction import build_proximity_graph, extract_connected_subgraphs

# Default paths (relative to repo root)
SIM_DATA = Path(
    "/scratch/pompeii/elferich/gouaux_tomo/ProcessingJE/solid_vs_liquid_2d/data"
)
STAR_FILES = {
    "Crystal": SIM_DATA / "crystal_aunp.star",
    "Liquid":  SIM_DATA / "liquid_resolved_aunp.star",
}

# Receptor star files (used for two-receptor mode)
RECEPTOR_STAR_FILES = {
    "Crystal": SIM_DATA / "crystal.star",
    "Liquid":  SIM_DATA / "liquid_resolved.star",
}

# Graph parameters (matching config.toml)
KDT_MAX_DISTANCE = 11.0   # nm
KDT_MIN_DISTANCE = 0.0    # nm
SUBGRAPH_SIZE = 4
NOISE_SIGMA_NM = 1.0       # Gaussian noise std (nm)
NOISE_SEED = 42

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Nimbus Sans L", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})


# ---------------------------------------------------------------------------
# Data loading and noise
# ---------------------------------------------------------------------------

def load_and_noise(star_path: Path, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Load AuNP coordinates (Å → nm) and add isotropic Gaussian noise."""
    df = starfile.read(star_path)
    coords_nm = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].values / 10.0
    noise = rng.normal(0.0, noise_sigma, size=coords_nm.shape)
    return coords_nm + noise


# ---------------------------------------------------------------------------
# Two-receptor subgraph extraction
# ---------------------------------------------------------------------------

def process_dataset_two_receptor(aunp_path: Path, label: str,
                                  noise_sigma: float, rng: np.random.Generator):
    """
    Build 4-AuNP subgraphs guaranteed to come from exactly two receptors.

    The AuNP star file is structured as:
      rows 0 … n_rec-1       → AuNP #1 per receptor (same order as receptor file)
      rows n_rec … 2*n_rec-1 → AuNP #2 per receptor (C2 mate)

    For every pair of receptors (i, j) where at least one AuNP from receptor i
    is within KDT_MAX_DISTANCE of at least one AuNP from receptor j, we form
    the 4-point subgraph [AuNP1_i, AuNP2_i, AuNP1_j, AuNP2_j].
    """
    print(f"\n=== {label} (two-receptor) ===")
    coords_nm = load_and_noise(aunp_path, noise_sigma, rng)
    n_rec = len(coords_nm) // 2
    aunp1 = coords_nm[:n_rec]   # (n_rec, 3)
    aunp2 = coords_nm[n_rec:]   # (n_rec, 3)
    print(f"  {n_rec} receptors, {len(coords_nm)} AuNP positions  (noise σ={noise_sigma} nm)")

    # Find receptor pairs where any inter-receptor AuNP distance <= KDT_MAX_DISTANCE.
    # Stack all AuNPs per receptor as (n_rec, 2, 3), then check pairwise.
    from scipy.spatial import cKDTree
    # Use AuNP#1 positions to find candidate receptor pairs efficiently,
    # then verify with all 4 inter-receptor distances.
    tree1 = cKDTree(aunp1)
    tree2 = cKDTree(aunp2)

    # All pairs within cutoff considering each combination of AuNP types
    pairs = set()
    for src, tgt in [(aunp1, tree1), (aunp1, tree2),
                     (aunp2, tree1), (aunp2, tree2)]:
        hits = tgt.query_ball_point(src, KDT_MAX_DISTANCE)
        for i, js in enumerate(hits):
            for j in js:
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

    print(f"  {len(pairs)} receptor pairs within {KDT_MAX_DISTANCE} nm")
    if not pairs:
        return np.array([]), np.array([])

    # Build subgraph tensor: (n_pairs, 4, 3)
    idx = list(pairs)
    i_arr = np.array([p[0] for p in idx])
    j_arr = np.array([p[1] for p in idx])
    subgraph_coords = np.stack([
        aunp1[i_arr], aunp2[i_arr],
        aunp1[j_arr], aunp2[j_arr],
    ], axis=1)  # (n_pairs, 4, 3)

    subgraphs = torch.tensor(subgraph_coords, dtype=torch.float32)
    features, feature_names = compute_pairwise_features(subgraphs)
    i_rot = feature_names.index("theta_rot")
    i_cc  = feature_names.index("theta_cc")
    theta_rot = features[:, i_rot]
    theta_cc  = features[:, i_cc]

    print(f"  theta_rot: mean={np.degrees(theta_rot).mean():.1f}°  "
          f"std={np.degrees(theta_rot).std():.1f}°")
    print(f"  theta_cc:  mean={np.degrees(theta_cc).mean():.1f}°  "
          f"std={np.degrees(theta_cc).std():.1f}°")
    return theta_rot, theta_cc


# ---------------------------------------------------------------------------
# Pipeline for one dataset
# ---------------------------------------------------------------------------

def process_dataset(star_path: Path, label: str, noise_sigma: float, rng: np.random.Generator):
    print(f"\n=== {label} ===")
    coords_nm = load_and_noise(star_path, noise_sigma, rng)
    print(f"  {len(coords_nm)} AuNP positions  (noise σ={noise_sigma} nm)")

    # Build graph and extract 4-node subgraphs
    graph, _ = build_proximity_graph(coords_nm, KDT_MAX_DISTANCE, KDT_MIN_DISTANCE)
    subgraphs, _ = extract_connected_subgraphs(graph, SUBGRAPH_SIZE, coords_nm)
    print(f"  {len(subgraphs)} connected {SUBGRAPH_SIZE}-node subgraphs")

    if len(subgraphs) == 0:
        return np.array([]), np.array([])

    # Compute pair-geometry features
    features, feature_names = compute_pairwise_features(subgraphs)
    i_rot = feature_names.index("theta_rot")
    i_cc  = feature_names.index("theta_cc")
    theta_rot = features[:, i_rot]
    theta_cc  = features[:, i_cc]

    print(f"  theta_rot: mean={np.degrees(theta_rot).mean():.1f}°  "
          f"std={np.degrees(theta_rot).std():.1f}°")
    print(f"  theta_cc:  mean={np.degrees(theta_cc).mean():.1f}°  "
          f"std={np.degrees(theta_cc).std():.1f}°")
    return theta_rot, theta_cc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str):
    ax.set_xlabel(r"$\theta_{\mathrm{rot}}$ ($\degree$)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\theta_{\mathrm{cc}}$ ($\degree$)", fontsize=15, fontweight="bold")
    ax.set_xlim(-95, 95)
    ax.set_ylim(-95, 95)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.6)
    ax.tick_params(labelsize=13, width=2.6, length=8)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontweight("bold")


def make_theta_scatter(datasets, output_path: Path):
    """
    datasets: list of (label, theta_rot_array, theta_cc_array)
    Two-panel hexbin density figure: one panel per dataset.
    """
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.2))
    if n == 1:
        axes = [axes]

    colors = plt.get_cmap("Dark2").colors
    for ax, (label, theta_rot, theta_cc), color in zip(axes, datasets, colors):
        tr_deg = np.degrees(theta_rot)
        tc_deg = np.degrees(theta_cc)
        ax.scatter(tr_deg, tc_deg, s=5, c=[color], alpha=0.15, linewidths=0,
                   rasterized=True, zorder=2)
        _style_ax(ax, label)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Theta scatter for simulated crystal vs liquid AuNP patterns")
    parser.add_argument("-o", "--output", default="output/sim_theta_scatter.png",
                        help="Output path (default: output/sim_theta_scatter.png)")
    parser.add_argument("--noise-sigma", type=float, default=NOISE_SIGMA_NM,
                        help=f"Gaussian noise std in nm (default: {NOISE_SIGMA_NM})")
    parser.add_argument("--seed", type=int, default=NOISE_SEED,
                        help=f"Random seed for noise (default: {NOISE_SEED})")
    parser.add_argument("--two-receptor", action="store_true",
                        help="Only use subgraphs from exactly two receptors")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    datasets = []
    for label, path in STAR_FILES.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        if args.two_receptor:
            tr, tc = process_dataset_two_receptor(path, label, args.noise_sigma, rng)
        else:
            tr, tc = process_dataset(path, label, args.noise_sigma, rng)
        if len(tr):
            datasets.append((label, tr, tc))

    if not datasets:
        print("No data found.")
        return

    make_theta_scatter(datasets, args.output)


if __name__ == "__main__":
    main()
