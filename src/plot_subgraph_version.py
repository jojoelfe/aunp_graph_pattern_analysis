#!/usr/bin/env python3
"""
Generate three individual SVG scatter plots:
  1. output/theta_scatter_all_mono.svg   — all real 4-node subgraphs (monochrome)
  2. output/sim_crystal_scatter.svg      — simulated crystal pattern
  3. output/sim_liquid_scatter.svg        — simulated liquid pattern

Usage:
    uv run python src/plot_subgraph_version.py
    uv run python src/plot_subgraph_version.py --noise-sigma 1.5
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import starfile
import torch

from alignment import compute_pairwise_features
from config import AnalysisConfig
from graph_construction import build_proximity_graph, extract_connected_subgraphs

# ---------------------------------------------------------------------------
# Simulation defaults (matching plot_simulated_theta_scatter.py)
# ---------------------------------------------------------------------------
SIM_DATA = Path(
    "/scratch/pompeii/elferich/gouaux_tomo/ProcessingJE/solid_vs_liquid_2d/data"
)
STAR_FILES = {
    "Crystal": SIM_DATA / "crystal_aunp.star",
    "Liquid":  SIM_DATA / "liquid_resolved_aunp.star",
}
NOISE_SIGMA_NM = 1.0
NOISE_SEED = 42

OUTPUT_DIR = Path("output")

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
# Shared axis styling
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str = None):
    ax.set_xlabel(r"$\theta_{\mathrm{rot}}$ ($\degree$)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\theta_{\mathrm{cc}}$ ($\degree$)", fontsize=15, fontweight="bold")
    ax.set_xlim(-95, 95)
    ax.set_ylim(-95, 95)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.6)
    ax.tick_params(labelsize=13, width=2.6, length=8)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontweight("bold")


# ---------------------------------------------------------------------------
# 1. All-mono experimental scatter
# ---------------------------------------------------------------------------

def load_experimental_data(config: AnalysisConfig):
    features = np.load(config.get_precomputed_path("features.npy"))
    with open(config.get_precomputed_path("feature_names.json")) as f:
        feature_names = json.load(f)
    with open(config.get_precomputed_path("metadata.json")) as f:
        metadata = json.load(f)
    return features, feature_names, metadata


# Subgraph annotations: global index -> display label
# (matching: --annotate 4540:1 4451:2 4465:3 4538:4)
ANNOTATIONS = {4540: "1", 4451: "2", 4465: "3", 4538: "4"}


def _build_tomo_labels(metadata):
    tomo_pack_info = metadata["tomo_pack_info"]
    tomo_names = metadata["tomo_names"].copy()
    extra_labels = ["Test Shape", "Liquid Decoy"]
    for i in range(len(tomo_pack_info) - len(tomo_names)):
        tomo_names.append(extra_labels[i] if i < len(extra_labels) else f"Extra {i}")
    color_idx = np.concatenate([np.full(n, i) for i, n in enumerate(tomo_pack_info)])
    tomo_labels = np.array([tomo_names[int(c)] for c in color_idx])
    skip = (tomo_labels == "Liquid Decoy") | (tomo_labels == "Test Shape")
    real_mask = ~skip
    return tomo_labels, real_mask


def plot_all_mono(features, feature_names, metadata, output_path: Path,
                  annotations: dict = None):
    if annotations is None:
        annotations = ANNOTATIONS

    i_rot = feature_names.index("theta_rot")
    i_cc = feature_names.index("theta_cc")
    theta_rot = np.degrees(features[:, i_rot])
    theta_cc = np.degrees(features[:, i_cc])

    _, real_mask = _build_tomo_labels(metadata)

    # Background: all real points except annotated ones
    bg_mask = real_mask.copy()
    for idx in annotations:
        if idx < len(bg_mask):
            bg_mask[idx] = False

    fig, ax = plt.subplots(figsize=(3.5, 3))
    color = plt.get_cmap("Dark2").colors[2]
    ax.scatter(theta_rot[bg_mask], theta_cc[bg_mask],
               s=15, c=[color], alpha=0.15, linewidths=0,
               rasterized=True, zorder=2)

    # Annotated subgraphs
    if annotations:
        ann_indices = sorted(annotations.keys())
        ann_arr = np.array(ann_indices)
        ax.scatter(theta_rot[ann_arr], theta_cc[ann_arr],
                   s=30, c="#aaaaaa", alpha=1.0, linewidths=1.2,
                   edgecolors="black", marker="o", rasterized=True, zorder=3)
        for idx in ann_indices:
            x, y = theta_rot[idx], theta_cc[idx]
            ax.annotate(
                annotations[idx],
                xy=(x, y),
                xytext=(x + 14, y),
                fontsize=7,
                ha="left",
                va="center",
                color="black",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.7,
                                shrinkA=3, shrinkB=2),
                annotation_clip=False,
                zorder=5,
            )

    _style_ax(ax, title="Experimental")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2 & 3. Simulated scatters (one file each)
# ---------------------------------------------------------------------------

def load_and_noise(star_path: Path, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    df = starfile.read(star_path)
    coords_nm = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].values / 10.0
    noise = rng.normal(0.0, noise_sigma, size=coords_nm.shape)
    return coords_nm + noise


def process_sim_dataset(star_path: Path, label: str, noise_sigma: float,
                        rng: np.random.Generator,
                        kdt_max_distance: float = 11.0,
                        kdt_min_distance: float = 0.0,
                        subgraph_size: int = 4):
    print(f"\n=== {label} ===")
    coords_nm = load_and_noise(star_path, noise_sigma, rng)
    print(f"  {len(coords_nm)} AuNP positions  (noise sigma={noise_sigma} nm)")

    graph, _ = build_proximity_graph(coords_nm, kdt_max_distance, kdt_min_distance)
    subgraphs, _ = extract_connected_subgraphs(graph, subgraph_size, coords_nm)
    print(f"  {len(subgraphs)} connected {subgraph_size}-node subgraphs")

    if len(subgraphs) == 0:
        return np.array([]), np.array([])

    features, feature_names = compute_pairwise_features(subgraphs)
    i_rot = feature_names.index("theta_rot")
    i_cc = feature_names.index("theta_cc")
    return features[:, i_rot], features[:, i_cc]


def plot_sim_scatter(theta_rot, theta_cc, label: str, color, output_path: Path):
    tr_deg = np.degrees(theta_rot)
    tc_deg = np.degrees(theta_cc)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.scatter(tr_deg, tc_deg, s=5, c=[color], alpha=0.15, linewidths=0,
               rasterized=True, zorder=2)
    _style_ax(ax, title=label)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate three individual SVG scatter plots")
    parser.add_argument("--noise-sigma", type=float, default=NOISE_SIGMA_NM,
                        help=f"Gaussian noise std in nm (default: {NOISE_SIGMA_NM})")
    parser.add_argument("--seed", type=int, default=NOISE_SEED,
                        help=f"Random seed for noise (default: {NOISE_SEED})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: output)")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Experimental all-mono scatter ---
    config_path = Path(os.environ.get("AUNP_CONFIG", "config.toml"))
    config = AnalysisConfig.from_toml_or_default(config_path)
    features, feature_names, metadata = load_experimental_data(config)
    plot_all_mono(features, feature_names, metadata, out / "theta_scatter_all_mono.svg")

    # --- 2 & 3. Simulated scatters ---
    rng = np.random.default_rng(args.seed)
    colors = plt.get_cmap("Dark2").colors
    sim_outputs = {
        "Crystal": out / "sim_crystal_scatter.svg",
        "Liquid":  out / "sim_liquid_scatter.svg",
    }

    for ci, (label, star_path) in enumerate(STAR_FILES.items()):
        if not star_path.exists():
            print(f"WARNING: {star_path} not found, skipping.")
            continue
        tr, tc = process_sim_dataset(star_path, label, args.noise_sigma, rng,
                                        kdt_max_distance=config.kdt_max_distance,
                                        kdt_min_distance=config.kdt_min_distance,
                                        subgraph_size=config.subgraph_size)
        if len(tr):
            plot_sim_scatter(tr, tc, label, colors[ci % len(colors)], sim_outputs[label])


if __name__ == "__main__":
    main()
