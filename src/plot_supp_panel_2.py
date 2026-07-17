#!/usr/bin/env python3
"""
Polished figure: Component size distribution (count of AUNPs) across all tomograms.

Recreates the diagnostic plot from diagnostics_h4kcys/component_size_distribution_aunps.png
with full control over styling for publication.

Usage:
    python src/plot_supp_panel_2.py
    python src/plot_supp_panel_2.py --config config.toml --output output/supp_panel_2.svg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import AnalysisConfig
from data_loading import load_all_coordinates
from graph_construction import build_proximity_graph

# ---------------------------------------------------------------------------
# Configuration — edit these to change the figure
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config.toml")
OUTPUT_PATH = Path("output/supp_panel_2.svg")

# Maximum component size shown as individual bar; larger sizes are grouped
MAX_SIZE = 12

# ---------------------------------------------------------------------------
# Figure styling — tweak these for the polished look
# ---------------------------------------------------------------------------
FIG_WIDTH = 4         # inches
FIG_HEIGHT = 3        # inches
DPI = 300

BAR_COLOR = "darkgreen"
BAR_ALPHA = 0.7
BAR_EDGE_COLOR = "black"

OVERFLOW_COLOR = "coral"
OVERFLOW_ALPHA = 0.7

TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 8

# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------

def make_figure(config_path: Path, output_path: Path):
    # 1. Load config and coordinates
    config = AnalysisConfig.from_toml_or_default(config_path)
    coordinate_list, tomo_names = load_all_coordinates(config)
    print(f"Loaded coordinates from {len(coordinate_list)} tomograms")

    # 2. Build graphs and collect component sizes per tomogram
    all_component_sizes = []
    for coords, name in zip(coordinate_list, tomo_names):
        graph, component_sizes = build_proximity_graph(
            coords.numpy(),
            max_distance=config.kdt_max_distance,
            min_distance=config.kdt_min_distance,
        )
        all_component_sizes.extend(component_sizes.tolist())
        print(f"  {name}: {len(component_sizes)} components, "
              f"{int(np.sum(component_sizes))} AUNPs")

    all_sizes = np.asarray(all_component_sizes)
    n_tomos = len(tomo_names)
    n_components = len(all_sizes)
    n_aunps = int(np.sum(all_sizes))
    print(f"Total: {n_tomos} tomograms, {n_components} components, {n_aunps} AUNPs")

    # 3. Compute histogram (count of AUNPs per component size)
    bins = np.arange(1, MAX_SIZE + 2) - 0.5
    counts, _ = np.histogram(all_sizes, bins=bins)
    x_pos = np.arange(1, MAX_SIZE + 1)
    y_values = counts * x_pos  # multiply by size to get AUNP count

    # Overflow: total AUNPs in components of size >= MAX_SIZE + 1
    n_overflow = int(np.sum(all_sizes[all_sizes >= MAX_SIZE + 1]))

    # 4. Create figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.bar(x_pos, y_values, color=BAR_COLOR, alpha=BAR_ALPHA,
           edgecolor=BAR_EDGE_COLOR)

    if n_overflow > 0:
        ax.bar([MAX_SIZE + 1], [n_overflow], color=OVERFLOW_COLOR,
               alpha=OVERFLOW_ALPHA, edgecolor=BAR_EDGE_COLOR,
               label=f"\u2265{MAX_SIZE + 1}")
        ax.legend(fontsize=TICK_FONTSIZE)

    ax.set_xlabel("Graph Size", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Count of AUNPs", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        f"All tomograms combined (Count of AUNPs)\n"
        f"({n_tomos} tomograms, {n_components} components, {n_aunps} AUNPs)",
        fontsize=TITLE_FONTSIZE, fontweight="bold",
    )
    ax.set_xticks(np.arange(1, MAX_SIZE + 2))
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xlim(0.5, MAX_SIZE + 1.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polished component-size distribution figure")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Path to TOML config file (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output file path (default: %(default)s)")
    args = parser.parse_args()

    make_figure(args.config, args.output)
