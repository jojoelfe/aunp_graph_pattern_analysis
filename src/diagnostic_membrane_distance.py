#!/usr/bin/env python3
"""
Diagnostic script: overlay circles on each AuNP in the active zonogram,
colored by distance to the postsynaptic membrane.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import mrcfile
from scipy.spatial import cKDTree

from config import AnalysisConfig
from data_loading import load_aunp_coordinates, load_membrane_mesh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_membrane_distances(
    coordinates: np.ndarray,
    membrane_positions: np.ndarray,
) -> np.ndarray:
    """Return distance from each AuNP to the nearest membrane vertex (nm)."""
    kdt = cKDTree(membrane_positions)
    distances, _ = kdt.query(coordinates)
    return distances


def plot_aunp_membrane_distance(
    coordinates: np.ndarray,
    distances: np.ndarray,
    active_zonogram_path: Path,
    output_path: Path,
    tomo_name: str,
    circle_radius: float = 8.0,
):
    """
    Plot the active zonogram with a circle around every AuNP,
    colored by its distance to the postsynaptic membrane.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # --- load zonogram and metadata ----------------------------------------
    with mrcfile.open(active_zonogram_path, permissive=True) as mrc:
        az_data = torch.tensor(mrc.data.copy())
    az_shape = az_data.shape  # (Z, Y, X)

    metadata_path = active_zonogram_path.with_suffix(".npy")
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    az_meta = np.load(metadata_path, allow_pickle=True).tolist()
    center = az_meta["center"]
    cs = az_meta["cs"]

    # Transform AuNP coords to zonogram pixel frame
    coords_px = (coordinates - center) @ cs.T
    coords_px += np.floor(np.array(az_shape)[[2, 1, 0]] / 2)

    # --- zonogram projection -----------------------------------------------
    projection = az_data.min(dim=0)[0].numpy()
    vmin, vmax = np.percentile(projection, [1, 99])
    ax.imshow(
        projection,
        cmap="gray",
        interpolation="mitchell",
        origin="lower",
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
    )

    # --- circles colored by distance ---------------------------------------
    norm = mcolors.Normalize(vmin=distances.min(), vmax=distances.max())
    cmap = plt.cm.plasma

    circles = []
    for i in range(len(coords_px)):
        c = Circle((coords_px[i, 0], coords_px[i, 1]), circle_radius)
        circles.append(c)

    col = PatchCollection(circles, cmap=cmap, norm=norm, alpha=0.7)
    col.set_array(distances)
    col.set_edgecolor("white")
    col.set_linewidth(1.2)
    ax.add_collection(col)

    # --- colorbar -----------------------------------------------------------
    cbar = fig.colorbar(col, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Distance to postsynaptic membrane (nm)", fontsize=11)

    # --- labels -------------------------------------------------------------
    ax.set_xlim(0, projection.shape[1])
    ax.set_ylim(0, projection.shape[0])
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.set_title(
        f"{tomo_name}\n{len(coordinates)} AuNPs — distance to membrane",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved membrane-distance overlay to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay circles on AuNPs colored by distance to postsynaptic membrane"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to configuration file (default: config.toml)",
    )
    args = parser.parse_args()

    config = AnalysisConfig.from_toml_or_default(args.config)

    for tomo in config.tomos:
        az_id = tomo[2] if len(tomo) == 3 else tomo[1]
        tomo_name = f"{tomo[0]}_{az_id}"
        logger.info(f"Processing {tomo_name} ...")

        # Load AuNP coordinates
        star_path = config.get_aunp_star_path(tomo)
        try:
            coordinates = load_aunp_coordinates(star_path)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {tomo_name}: {e}")
            continue

        # Load membrane mesh
        glb_path = config.get_membrane_glb_path(tomo)
        try:
            mem_positions, _ = load_membrane_mesh(glb_path)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {tomo_name}: {e}")
            continue

        # Compute per-AuNP distance to membrane
        distances = compute_membrane_distances(coordinates, mem_positions)
        logger.info(
            f"  distances — min: {distances.min():.1f} nm, "
            f"max: {distances.max():.1f} nm, mean: {distances.mean():.1f} nm"
        )

        # Load active zonogram and create overlay
        az_path = config.get_active_zonogram_path(tomo)
        if not az_path.exists():
            logger.warning(f"Active zonogram not found: {az_path}, skipping")
            continue

        output_path = config.diagnostics_folder / f"{tomo_name}_membrane_distance.png"
        plot_aunp_membrane_distance(
            coordinates, distances, az_path, output_path, tomo_name
        )


if __name__ == "__main__":
    main()
