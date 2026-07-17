#!/usr/bin/env python3
"""
Polished figure: AuNP graph with highlighted subgraphs for a single tomogram.

Recreates the diagnostic plot from diagnostics_h4kcys/AMmilled55-1_Position_10_3_0_graph_subgraphs.png
with full control over styling for publication.

Usage:
    python src/plot_supp_panel_1.py
    python src/plot_supp_panel_1.py --output figures/my_figure.png
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mrcfile
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from data_loading import load_aunp_coordinates, load_active_zonogram_metadata
from graph_construction import build_proximity_graph, extract_connected_subgraphs

# ---------------------------------------------------------------------------
# Configuration — edit these to change the figure
# ---------------------------------------------------------------------------

# Tomogram identity
BASEFOLDER = Path("/nrs/elferich/gouaux_tomo/15F1-H4K2Cys/TOP_TOMOS")
TOMO_NAME = "AMmilled55-1_Position_10_3"
ALIGNMENT_DIR = "liza_az1"
AZ_ID = "0"

# Graph parameters (must match the precompute run)
KDT_MAX_DISTANCE = 13.0   # nm
KDT_MIN_DISTANCE = 0.0    # nm
SUBGRAPH_SIZE = 4

# How many subgraphs to highlight (evenly spaced through the list)
N_HIGHLIGHT = 2

# Output
OUTPUT_PATH = Path("output/supp_panel_1.svg")

# ---------------------------------------------------------------------------
# Figure styling — tweak these for the polished look
# ---------------------------------------------------------------------------
FIG_WIDTH = 8        # inches
FIG_HEIGHT = 6.5     # inches
DPI = 300

# Background tomogram
BG_CMAP = "gray"
BG_ALPHA = 1.0
BG_CONTRAST_PCTL = (1, 99)   # percentile stretch

# All-AuNP scatter (unfilled light blue circles)
AUNP_COLOR = "none"
AUNP_EDGE_COLOR = "lightskyblue"
AUNP_SIZE = 105
AUNP_EDGE_WIDTH = 1.2
AUNP_ALPHA = 0.0

# Graph edges (non-highlighted)
EDGE_COLOR = "steelblue"
EDGE_ALPHA = 0.8
EDGE_WIDTH = 2.0

# Highlighted subgraphs
HIGHLIGHT_COLORS = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # purple
    "#e7298a",  # pink
    "#66a61e",  # green
]
HIGHLIGHT_NODE_SIZE = 90
HIGHLIGHT_NODE_EDGE_WIDTH = 1.5
HIGHLIGHT_NODE_ALPHA = 0.7
HIGHLIGHT_EDGE_WIDTH = 2.5
HIGHLIGHT_EDGE_ALPHA = 0.6
HIGHLIGHT_JITTER_SIGMA = 0.5  # pixels, for visual separation of overlapping subgraphs

# Title / labels
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 8

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _resolve_star_path() -> Path:
    """Find the best available STAR file (manual_refined > manual > plain)."""
    aunps_dir = BASEFOLDER / TOMO_NAME / ALIGNMENT_DIR / "aunps"
    for suffix in ("_manual_refined", "_manual", ""):
        candidate = aunps_dir / f"aunp_tm_BP_active_zone_{AZ_ID}{suffix}.star"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No STAR file found in {aunps_dir}")


def _load_active_zonogram():
    """Load the active zonogram projection and coordinate-system metadata."""
    az_mrc_path = BASEFOLDER / TOMO_NAME / ALIGNMENT_DIR / "active_zonograms" / f"active_zonogram_{AZ_ID}.mrc"
    az_npy_path = az_mrc_path.with_suffix(".npy")

    with mrcfile.open(az_mrc_path, permissive=True) as mrc:
        az_data = torch.tensor(mrc.data.copy())

    az_meta = np.load(az_npy_path, allow_pickle=True).tolist()
    az_center = az_meta["center"]
    az_cs = az_meta["cs"]
    az_offset = np.floor(np.array(az_data.shape)[[2, 1, 0]] / 2)

    # Min-projection along Z for a 2-D image
    projection = az_data.min(dim=0)[0].numpy()

    def nm_to_plot(coords):
        """Convert nm coordinates to active-zonogram pixel coordinates."""
        return (coords - az_center) @ az_cs.T + az_offset

    return projection, nm_to_plot

# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------

def make_figure():
    # 1. Load coordinates
    star_path = _resolve_star_path()
    coordinates = load_aunp_coordinates(star_path)
    print(f"Loaded {len(coordinates)} AuNPs from {star_path.name}")

    # 2. Build graph
    graph, component_sizes = build_proximity_graph(
        coordinates,
        max_distance=KDT_MAX_DISTANCE,
        min_distance=KDT_MIN_DISTANCE,
    )

    # 3. Extract subgraphs
    subgraphs, _ = extract_connected_subgraphs(graph, SUBGRAPH_SIZE, coordinates)
    print(f"Found {len(subgraphs)} connected subgraphs of size {SUBGRAPH_SIZE}")

    # 4. Load active zonogram + coordinate transform
    projection, nm_to_plot = _load_active_zonogram()

    # Transform all coordinates to plot space
    plot_coords = nm_to_plot(coordinates)
    transformed_subgraphs = [nm_to_plot(sg.numpy()) for sg in subgraphs]

    # 5. Create figure
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Background tomogram
    vmin, vmax = np.percentile(projection, BG_CONTRAST_PCTL)
    ax.imshow(
        projection,
        cmap=BG_CMAP,
        interpolation="mitchell",
        origin="lower",
        alpha=BG_ALPHA,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(0, projection.shape[1])
    ax.set_ylim(0, projection.shape[0])

    # All edges
    edge_list = graph.edge_list()
    for i, j in edge_list:
        p1, p2 = plot_coords[i], plot_coords[j]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=EDGE_COLOR, alpha=EDGE_ALPHA, linewidth=EDGE_WIDTH, zorder=1,
        )

    # All AuNPs
    ax.scatter(
        plot_coords[:, 0], plot_coords[:, 1],
        c=AUNP_COLOR, s=AUNP_SIZE, alpha=AUNP_ALPHA,
        edgecolors=AUNP_EDGE_COLOR, linewidth=AUNP_EDGE_WIDTH,
        label="All AUNPs", zorder=2,
    )

    # Highlighted subgraphs
    n_show = min(N_HIGHLIGHT, len(transformed_subgraphs))
    rng = np.random.default_rng(14)
    indices = rng.choice(len(transformed_subgraphs), n_show, replace=False)

    for k, sg_idx in enumerate(indices):
        sg = transformed_subgraphs[sg_idx]
        color = HIGHLIGHT_COLORS[k % len(HIGHLIGHT_COLORS)]
        jitter = rng.normal(0, HIGHLIGHT_JITTER_SIGMA, size=2)

        # Edges within subgraph (fully connected display)
        for i in range(len(sg)):
            for j in range(i + 1, len(sg)):
                ax.plot(
                    [sg[i, 0] + jitter[0], sg[j, 0] + jitter[0]],
                    [sg[i, 1] + jitter[1], sg[j, 1] + jitter[1]],
                    color=color, alpha=HIGHLIGHT_EDGE_ALPHA,
                    linewidth=HIGHLIGHT_EDGE_WIDTH, zorder=9,
                )

        # Nodes
        ax.scatter(
            sg[:, 0] + jitter[0], sg[:, 1] + jitter[1],
            c=color, s=HIGHLIGHT_NODE_SIZE,
            edgecolors="black", linewidth=HIGHLIGHT_NODE_EDGE_WIDTH,
            alpha=HIGHLIGHT_NODE_ALPHA, zorder=10,
        )

    # Labels and title
    ax.set_xlabel("X (pixels)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Y (pixels)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_aspect("equal")

    tomo_label = f"{TOMO_NAME}_{AZ_ID}"
    ax.set_title(
        f"{tomo_label}\n"
        f"{len(coordinates)} AUNPs, {len(subgraphs)} subgraphs, "
        f"{len(edge_list)} edges",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )

    ax.legend(loc="upper right", framealpha=0.8, fontsize=8)

    fig.tight_layout(pad=0.5)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Polished AuNP subgraph figure")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output file path (default: %(default)s)")
    args = parser.parse_args()

    OUTPUT_PATH = args.output
    make_figure()
