#!/usr/bin/env python
"""Generate a high-quality scatter plot of theta_rot vs theta_cc.

Usage:
    uv run python src/plot_theta_scatter.py
    uv run python src/plot_theta_scatter.py -o figures/theta_scatter.png
    AUNP_CONFIG=config_h4kcys.toml uv run python src/plot_theta_scatter.py
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import AnalysisConfig


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Nimbus Sans L", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})


def load_data(config: AnalysisConfig):
    """Load features and metadata from precomputed files."""
    features = np.load(config.get_precomputed_path("features.npy"))
    with open(config.get_precomputed_path("feature_names.json")) as f:
        feature_names = json.load(f)
    with open(config.get_precomputed_path("metadata.json")) as f:
        metadata = json.load(f)
    return features, feature_names, metadata


def make_highlight_component_map(config: AnalysisConfig, tomo_name: str,
                                  labeled_subgraphs: dict, output_path: Path):
    """Re-generate the component_size diagnostic for one tomogram with numbered labels.

    labeled_subgraphs: {label_number: np.ndarray (subgraph_size, 3) in nm}
    """
    from data_loading import load_aunp_coordinates
    from graph_construction import build_proximity_graph, extract_connected_subgraphs
    from visualization import plot_graph_diagnostics

    tomo = next((t for t in config.tomos if f"{t[0]}_{t[2]}" == tomo_name), None)
    if tomo is None:
        print(f"Warning: tomogram {tomo_name!r} not found in config")
        return

    star_path = config.get_aunp_star_path(tomo)
    coords_nm = load_aunp_coordinates(star_path)

    graph, _ = build_proximity_graph(
        coords_nm, config.kdt_max_distance, config.kdt_min_distance
    )
    subgraphs_tensor, _ = extract_connected_subgraphs(graph, config.subgraph_size, coords_nm)

    az_path = config.get_active_zonogram_path(tomo)
    plot_graph_diagnostics(
        coordinates=coords_nm,
        graph=graph,
        subgraphs=subgraphs_tensor,
        tomogram_name=tomo_name,
        output_path=output_path,
        active_zonogram_path=az_path,
        plot_type="component_size",
        labeled_subgraphs=labeled_subgraphs,
    )
    print(f"Saved labeled component map: {output_path}")


def make_theta_scatter(features, feature_names, metadata, output_path: Path,
                       fold: bool = False, overlay: bool = False,
                       highlight_tomo: str = None, config: AnalysisConfig = None):
    """Create publication-quality scatter plot of theta_rot vs theta_cc."""
    # Resolve feature indices
    i_rot = feature_names.index("theta_rot")
    i_cc = feature_names.index("theta_cc")
    theta_rot = np.degrees(features[:, i_rot]).copy()
    theta_cc = np.degrees(features[:, i_cc]).copy()

    # Canonical fold: the pair-label swap symmetry maps
    #   (theta_rot, theta_cc) -> (-theta_rot, theta_cc - theta_rot)
    # We fold into the theta_rot >= 0 half.
    if fold:
        need_fold = theta_rot < 0
        theta_cc[need_fold] = theta_cc[need_fold] - theta_rot[need_fold]
        theta_rot[need_fold] = -theta_rot[need_fold]

    # Overlay: duplicate each point with its pair-swap equivalent
    if overlay:
        theta_rot_swap = -theta_rot
        theta_cc_swap = theta_cc - theta_rot
        theta_rot = np.concatenate([theta_rot, theta_rot_swap])
        theta_cc = np.concatenate([theta_cc, theta_cc_swap])

    # Reconstruct per-subgraph tomogram assignment
    tomo_pack_info = metadata["tomo_pack_info"]
    tomo_names = metadata["tomo_names"].copy()
    extra_labels = ["Test Shape", "Liquid Decoy"]
    for i in range(len(tomo_pack_info) - len(tomo_names)):
        tomo_names.append(extra_labels[i] if i < len(extra_labels) else f"Extra {i}")

    color_idx = np.concatenate([np.full(n, i) for i, n in enumerate(tomo_pack_info)])
    tomo_labels = np.array([tomo_names[int(c)] for c in color_idx])

    # Filter out decoys and test shapes — only real tomogram subgraphs
    skip = (tomo_labels == "Liquid Decoy") | (tomo_labels == "Test Shape")
    real_mask = ~skip

    comp_sizes = np.array(metadata.get("component_sizes", []))
    subgraph_size = metadata.get("subgraph_size", 4)
    if len(comp_sizes) == len(features):
        is_exact = comp_sizes == subgraph_size
    else:
        is_exact = np.zeros(len(features), dtype=bool)

    # Duplicate metadata arrays for overlay mode
    if overlay:
        tomo_labels = np.concatenate([tomo_labels, tomo_labels])
        real_mask = np.concatenate([real_mask, real_mask])
        is_exact = np.concatenate([is_exact, is_exact])

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Real subgraphs per tomogram
    unique_tomos = sorted(set(tomo_labels[real_mask]))

    labeled_subgraphs = {}  # {label_num: coords_nm (subgraph_size, 3)} for component map

    if highlight_tomo is not None:
        # Grey background, red highlight
        for tname in unique_tomos:
            mask = real_mask & (tomo_labels == tname)
            exact = mask & is_exact
            regular = mask & ~is_exact
            is_hl = tname == highlight_tomo
            color = "red" if is_hl else "#aaaaaa"
            alpha_reg = 0.4 if is_hl else 0.15
            zorder = 4 if is_hl else 2

            if regular.any():
                ax.scatter(theta_rot[regular], theta_cc[regular],
                           s=15, c=color, alpha=alpha_reg, linewidths=0.3,
                           edgecolors="none", marker="o", rasterized=True,
                           label=tname if is_hl else None, zorder=zorder)
            if exact.any():
                ax.scatter(theta_rot[exact], theta_cc[exact],
                           s=35, c=color, alpha=1.0, linewidths=0.5,
                           edgecolors="white", marker="D", rasterized=True,
                           label=f"{tname} (exact)" if is_hl and not regular.any() else None,
                           zorder=zorder + 1)

            # Number the highlighted tomo's exact diamonds and collect coords
            if is_hl and exact.any() and config is not None and not overlay:
                sg_original = torch.load(
                    config.get_precomputed_path("subgraphs_original.pt"),
                    weights_only=True
                )
                exact_global_indices = np.where(exact)[0]
                for label_num, global_idx in enumerate(exact_global_indices):
                    ax.annotate(
                        str(label_num),
                        xy=(theta_rot[global_idx], theta_cc[global_idx]),
                        fontsize=6, ha="center", va="center",
                        color="white", fontweight="bold", zorder=zorder + 3,
                        annotation_clip=True,
                    )
                    labeled_subgraphs[label_num] = sg_original[global_idx].numpy()
    else:
        # Bold qualitative palette from Matplotlib
        colors = plt.get_cmap("Dark2").colors
        for ti, tname in enumerate(unique_tomos):
            mask = real_mask & (tomo_labels == tname)
            exact = mask & is_exact
            regular = mask & ~is_exact

            color = colors[ti % len(colors)]

            if regular.any():
                ax.scatter(theta_rot[regular], theta_cc[regular],
                           s=15, c=[color], alpha=0.2, linewidths=0.3,
                           edgecolors="white", marker="o", rasterized=True,
                           label=tname, zorder=2)
            if exact.any():
                ax.scatter(theta_rot[exact], theta_cc[exact],
                           s=35, c=[color], alpha=1.0, linewidths=0.5,
                           edgecolors="white", marker="D", rasterized=True,
                           label=f"{tname} (exact)" if not regular.any() else None,
                           zorder=3)

    ax.set_xlabel(r"$\theta_{\mathrm{rot}}$ ($\degree$)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\theta_{\mathrm{cc}}$ ($\degree$)", fontsize=15, fontweight="bold")
    if fold:
        ax.set_xlim(-5, 185)
        cc_min = np.floor(theta_cc[real_mask].min() / 10) * 10 - 5
        cc_max = np.ceil(theta_cc[real_mask].max() / 10) * 10 + 5
        ax.set_ylim(cc_min, cc_max)
    elif overlay:
        cc_min = np.floor(theta_cc[real_mask].min() / 10) * 10 - 5
        cc_max = np.ceil(theta_cc[real_mask].max() / 10) * 10 + 5
        ax.set_xlim(-185, 185)
        ax.set_ylim(cc_min, cc_max)
    else:
        ax.set_xlim(-95, 95)
        ax.set_ylim(-95, 95)
    for spine in ax.spines.values():
        spine.set_linewidth(2.6)

    ax.tick_params(labelsize=13, width=2.6, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)

    # Generate companion labeled component map
    if highlight_tomo is not None and labeled_subgraphs and config is not None:
        map_path = output_path.parent / f"{output_path.stem}_{highlight_tomo}_component_map.png"
        make_highlight_component_map(config, highlight_tomo, labeled_subgraphs, map_path)


def _build_tomo_labels(metadata):
    """Return (tomo_labels array, real_mask, is_exact) for all subgraphs."""
    tomo_pack_info = metadata["tomo_pack_info"]
    tomo_names = metadata["tomo_names"].copy()
    extra_labels = ["Test Shape", "Liquid Decoy"]
    for i in range(len(tomo_pack_info) - len(tomo_names)):
        tomo_names.append(extra_labels[i] if i < len(extra_labels) else f"Extra {i}")

    color_idx = np.concatenate([np.full(n, i) for i, n in enumerate(tomo_pack_info)])
    tomo_labels = np.array([tomo_names[int(c)] for c in color_idx])

    skip = (tomo_labels == "Liquid Decoy") | (tomo_labels == "Test Shape")
    real_mask = ~skip
    return tomo_labels, real_mask, tomo_names


def _resolve_tomo_name(tomo_names, query):
    """Find the tomo name that contains *query* as a substring (case-insensitive)."""
    q = query.lower()
    matches = [n for n in tomo_names if q in n.lower()]
    if not matches:
        return query  # fall back to exact
    return matches[0]


def _ax_style(ax):
    ax.set_xlabel(r"$\theta_{\mathrm{rot}}$ ($\degree$)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\theta_{\mathrm{cc}}$ ($\degree$)", fontsize=15, fontweight="bold")
    ax.set_xlim(-95, 95)
    ax.set_ylim(-95, 95)
    for spine in ax.spines.values():
        spine.set_linewidth(2.6)
    ax.tick_params(labelsize=13, width=2.6, length=8)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontweight("bold")


def make_theta_scatter_exact4(features, feature_names, metadata, output_path: Path,
                               highlight_tomo: str = "Position_10_3"):
    """Black scatter of exact-4-component subgraphs; label highlight_tomo points on the side."""
    i_rot = feature_names.index("theta_rot")
    i_cc = feature_names.index("theta_cc")
    theta_rot = np.degrees(features[:, i_rot])
    theta_cc = np.degrees(features[:, i_cc])

    tomo_labels, real_mask, tomo_names = _build_tomo_labels(metadata)
    full_tomo = _resolve_tomo_name(tomo_names, highlight_tomo)

    comp_sizes = np.array(metadata.get("component_sizes", []))
    subgraph_size = metadata.get("subgraph_size", 4)
    is_exact = (comp_sizes == subgraph_size) if len(comp_sizes) == len(features) \
        else np.zeros(len(features), dtype=bool)

    exact_real = real_mask & is_exact

    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Which of the 4 highlight points to show and how to label them
    # Keys are 0-based positions within the highlight set; values are display labels.
    show_positions = {0: "2", 1: "3", 3: "4"}

    hl_indices = np.where(exact_real & (tomo_labels == full_tomo))[0]
    shown_global = [hl_indices[p] for p in show_positions if p < len(hl_indices)]
    shown_labels = [show_positions[p] for p in show_positions if p < len(hl_indices)]
    shown_set = set(shown_global)

    # Background: all exact-real points not in the shown highlighted set
    bg_mask = exact_real.copy()
    for gi in shown_global:
        bg_mask[gi] = False

    ax.scatter(theta_rot[bg_mask], theta_cc[bg_mask],
               s=35, c="black", alpha=0.4, linewidths=0,
               marker="o", rasterized=True, zorder=2)

    # Shown highlighted points: grey with black outline + side label
    if shown_global:
        shown_arr = np.array(shown_global)
        ax.scatter(theta_rot[shown_arr], theta_cc[shown_arr],
                   s=35, c="#aaaaaa", alpha=1.0, linewidths=1.2,
                   edgecolors="black", marker="o", rasterized=True, zorder=3)

        for idx, label_str in zip(shown_global, shown_labels):
            x, y = theta_rot[idx], theta_cc[idx]
            ax.annotate(
                label_str,
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

    _ax_style(ax)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def make_theta_scatter_all_mono(features, feature_names, metadata, output_path: Path,
                                annotations: dict = None):
    """Monochrome scatter of all real 4-node subgraphs, styled like sim_theta_scatter.

    Args:
        annotations: Optional dict mapping global subgraph index -> display label string.
                     Annotated points are highlighted with a side label and leader line.
    """
    i_rot = feature_names.index("theta_rot")
    i_cc = feature_names.index("theta_cc")
    theta_rot = np.degrees(features[:, i_rot])
    theta_cc = np.degrees(features[:, i_cc])

    tomo_labels, real_mask, _ = _build_tomo_labels(metadata)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Background points (exclude annotated ones so they don't double-draw)
    bg_mask = real_mask.copy()
    if annotations:
        for idx in annotations:
            if idx < len(bg_mask):
                bg_mask[idx] = False

    ax.scatter(theta_rot[bg_mask], theta_cc[bg_mask],
               s=15, c="black", alpha=0.15, linewidths=0,
               rasterized=True, zorder=2)

    # Annotated points
    if annotations:
        ann_indices = sorted(annotations.keys())
        ann_arr = np.array(ann_indices)
        ax.scatter(theta_rot[ann_arr], theta_cc[ann_arr],
                   s=30, c="#aaaaaa", alpha=1.0, linewidths=1.2,
                   edgecolors="black", marker="o", rasterized=True, zorder=3)

        for idx in ann_indices:
            label_str = annotations[idx]
            x, y = theta_rot[idx], theta_cc[idx]
            ax.annotate(
                label_str,
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

    _ax_style(ax)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scatter plot of theta_rot vs theta_cc")
    parser.add_argument("-o", "--output", default="output/theta_scatter.png",
                        help="Output file path (default: output/theta_scatter.png)")
    parser.add_argument("--fold", action="store_true",
                        help="Apply canonical pair-swap fold (theta_rot >= 0)")
    parser.add_argument("--overlay", action="store_true",
                        help="Plot both pair-swap representations of each point")
    parser.add_argument("--highlight", metavar="TOMO",
                        help="Highlight points from this tomogram in red; all others grey")
    parser.add_argument("--exact4", action="store_true",
                        help="Plot only exact-4-component subgraphs (black, side-labeled)")
    parser.add_argument("--all-mono", action="store_true",
                        help="Plot all real subgraphs monochrome (sim_theta_scatter style)")
    parser.add_argument("--label-tomo", default="Position_10_3",
                        help="Tomogram substring to label in --exact4 plot (default: Position_10_3)")
    parser.add_argument("--annotate", nargs="+", metavar="IDX:LABEL",
                        help="Annotate specific subgraphs in --all-mono plot. "
                             "Format: IDX:LABEL (e.g. 245:1 247:2)")
    args = parser.parse_args()

    config_path = Path(os.environ.get("AUNP_CONFIG", "config.toml"))
    config = AnalysisConfig.from_toml_or_default(config_path)

    features, feature_names, metadata = load_data(config)

    if args.exact4:
        stem = Path(args.output).stem
        out = Path(args.output).parent / f"{stem}_exact4.png"
        make_theta_scatter_exact4(features, feature_names, metadata, out,
                                   highlight_tomo=args.label_tomo)
    elif args.all_mono:
        stem = Path(args.output).stem
        out = Path(args.output).parent / f"{stem}_all_mono.png"
        annotations = None
        if args.annotate:
            annotations = {}
            for spec in args.annotate:
                idx_str, label = spec.split(":", 1)
                annotations[int(idx_str)] = label
        make_theta_scatter_all_mono(features, feature_names, metadata, out,
                                   annotations=annotations)
    else:
        make_theta_scatter(features, feature_names, metadata, args.output,
                           fold=args.fold, overlay=args.overlay,
                           highlight_tomo=args.highlight, config=config)
