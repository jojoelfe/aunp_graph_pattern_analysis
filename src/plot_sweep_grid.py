#!/usr/bin/env python3
"""
Combined grid figures for the parameter sweep (paired analysis).

Two modes:
  --sweep distance  Rows = kdt_max_distance values at a fixed noise sigma.
                    Columns = [Experimental | Crystal | Liquid].
  --sweep noise     Rows = simulated noise sigma values at a fixed distance.
                    Columns = [Crystal | Liquid] (Experimental omitted since
                    it does not depend on the simulated noise level).

Reuses the precomputed experimental data in
output_paired_sweep/precomputed_dist{D} and re-derives the simulated
crystal/liquid scatters with the paired analysis (same logic as
plot_STA_paired_version.py).

Usage:
    uv run python src/plot_sweep_grid.py --sweep distance --distances 10 12 14 16 --noise-sigma 1.0
    uv run python src/plot_sweep_grid.py --sweep noise --distance 12 --noise-values 0.8 1.0 1.2 1.4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import AnalysisConfig
from precompute_paired import (
    build_ampa_proximity_graph,
    compute_features_known_pairing,
    extract_ampa_pairs,
    make_subgraphs_from_ampa_pairs,
)
from plot_STA_paired_version import (
    load_and_noise_paired,
    load_experimental_data,
    _build_tomo_labels,
)

SWEEP_DIR = Path("output_paired_sweep")
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


def process_sim_dataset_paired(star_path: Path, noise_sigma: float,
                                rng: np.random.Generator, kdt_max_distance: float):
    ampas = load_and_noise_paired(star_path, noise_sigma, rng)
    edges = build_ampa_proximity_graph(ampas, kdt_max_distance)
    ampa_pairs, _ = extract_ampa_pairs(ampas, edges)
    if len(ampa_pairs) == 0:
        return np.array([]), np.array([])
    subgraphs = make_subgraphs_from_ampa_pairs(ampas, ampa_pairs)
    features, feature_names = compute_features_known_pairing(subgraphs)
    i_rot = feature_names.index("theta_rot")
    i_cc = feature_names.index("theta_cc")
    return features[:, i_rot], features[:, i_cc]


def _style_ax(ax):
    ax.set_xlim(-95, 95)
    ax.set_ylim(-95, 95)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(labelsize=11, width=2.0, length=6)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontweight("bold")


def make_grid(rows, include_experimental: bool, output_path: Path):
    """
    rows: list of (row_label, distance, noise_sigma) tuples, one per grid row.
    include_experimental: whether to add an Experimental column (col 0).
    """
    columns = ["Experimental", "Crystal", "Liquid"] if include_experimental \
        else ["Crystal", "Liquid"]
    n_rows = len(rows)
    n_cols = len(columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.0 * n_rows),
                              sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    colors = plt.get_cmap("Dark2").colors
    exp_color = colors[2]
    sim_colors = {"Crystal": colors[0], "Liquid": colors[1]}
    sim_col_offset = 1 if include_experimental else 0

    for row, (row_label, dist, noise_sigma) in enumerate(rows):
        config_path = SWEEP_DIR / f"config_dist{dist}.toml"
        config = AnalysisConfig.from_toml(config_path)

        if include_experimental:
            features, feature_names, metadata = load_experimental_data(config)
            i_rot = feature_names.index("theta_rot")
            i_cc = feature_names.index("theta_cc")
            theta_rot = np.degrees(features[:, i_rot])
            theta_cc = np.degrees(features[:, i_cc])
            _, real_mask = _build_tomo_labels(metadata)

            ax = axes[row, 0]
            ax.scatter(theta_rot[real_mask], theta_cc[real_mask],
                       s=15, c=[exp_color], alpha=0.3, linewidths=0,
                       rasterized=True, zorder=2)
            _style_ax(ax)

        # Simulated Crystal / Liquid (paired, fresh rng per row)
        rng = np.random.default_rng(NOISE_SEED)
        star_files = {
            "Crystal": Path(config.sim_crystal_star_file),
            "Liquid": Path(config.sim_liquid_star_file),
        }
        for col_i, label in enumerate(["Crystal", "Liquid"]):
            star_path = star_files[label]
            ax = axes[row, col_i + sim_col_offset]
            if not star_path.exists():
                print(f"WARNING: {star_path} not found, skipping.")
                _style_ax(ax)
                continue
            tr, tc = process_sim_dataset_paired(
                star_path, noise_sigma, rng, kdt_max_distance=config.kdt_max_distance
            )
            if len(tr):
                ax.scatter(np.degrees(tr), np.degrees(tc),
                           s=15, c=[sim_colors[label]], alpha=0.3, linewidths=0,
                           rasterized=True, zorder=2)
            _style_ax(ax)

        # Row label on the far left
        axes[row, 0].set_ylabel(
            row_label + "\n" + r"$\theta_{\mathrm{cc}}$ ($\degree$)",
            fontsize=12, fontweight="bold"
        )

    # Column titles on top row only
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=15, fontweight="bold")

    # X labels on bottom row only
    for col in range(n_cols):
        axes[-1, col].set_xlabel(r"$\theta_{\mathrm{rot}}$ ($\degree$)",
                                  fontsize=13, fontweight="bold")

    fig.subplots_adjust(left=0.16 if not include_experimental else 0.13,
                        right=0.995, top=0.95, bottom=0.075,
                        wspace=0.05, hspace=0.05)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    if output_path.suffix != ".png":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved {png_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Combined parameter-sweep grid (paired analysis)")
    parser.add_argument("--sweep", choices=["distance", "noise"], default="distance",
                        help="Which parameter varies across rows (default: distance)")
    parser.add_argument("--distances", type=int, nargs="+", default=[10, 12, 14, 16],
                        help="Distance values for --sweep distance (default: 10 12 14 16)")
    parser.add_argument("--noise-sigma", type=float, default=1.0,
                        help="Fixed simulated noise std (nm) for --sweep distance (default: 1.0)")
    parser.add_argument("--distance", type=int, default=12,
                        help="Fixed distance (nm) for --sweep noise (default: 12)")
    parser.add_argument("--noise-values", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4],
                        help="Noise sigma values for --sweep noise (default: 0.8 1.0 1.2 1.4)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (default depends on --sweep)")
    args = parser.parse_args()

    if args.sweep == "distance":
        rows = [(rf"$d_{{\mathrm{{min}}}}$ < {d} nm", d, args.noise_sigma)
                for d in args.distances]
        output = args.output or f"output_paired_sweep/sweep_grid_dist_noise{args.noise_sigma}.svg"
        make_grid(rows, include_experimental=True, output_path=Path(output))
    else:
        rows = [(rf"$\sigma$ = {n} nm", args.distance, n) for n in args.noise_values]
        output = args.output or f"output_paired_sweep/sweep_grid_noise_dist{args.distance}.svg"
        make_grid(rows, include_experimental=False, output_path=Path(output))


if __name__ == "__main__":
    main()
