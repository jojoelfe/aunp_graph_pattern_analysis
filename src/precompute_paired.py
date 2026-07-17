#!/usr/bin/env python3
"""
Precompute paired AuNP analysis.

Uses a star file with known AMPA-AuNP pairing to build subgraphs
where the pairing is determined by biology (same AMPA receptor)
rather than minimum-distance matching.

Pipeline:
1. Load paired AuNP star file (all tomograms in one file)
2. Group AuNPs into AMPA receptors (each with 2 AuNPs)
3. Build AMPA-level proximity graph per tomogram
4. Extract connected AMPA pairs → 4-AuNP subgraphs with known pairing
5. Align to membrane normal
6. Compute pair-geometry features using known pairing
7. Generate plots

Usage:
    uv run python src/precompute_paired.py
    uv run python src/precompute_paired.py --config config_paired.toml --force
"""

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import einops
import numpy as np
import starfile
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from alignment import _canonicalize_direction, _signed_angle_from_to
from config import AnalysisConfig
from data_loading import load_membrane_mesh, save_precomputed_data
from graph_construction import align_subgraphs_to_membrane

logger = logging.getLogger(__name__)


def load_paired_star_file(star_path: Path) -> dict:
    """Load paired AuNP star file and group by tomogram and AMPA.

    Returns:
        dict mapping tomo_name -> list of AMPAs, where each AMPA is:
            {"ampa_id": int, "aunp1": (x, y, z), "aunp2": (x, y, z)}
    """
    raw = starfile.read(star_path)
    # starfile returns a dict when multiple data blocks exist
    data = raw["particles"] if isinstance(raw, dict) else raw

    coord_cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    tomo_col = "rlnTomoName"
    ampa_col = "faAMPA_ID"
    type_col = "faAuNP_Type"

    tomo_ampas = {}

    for tomo_name, group in data.groupby(tomo_col):
        ampas = []
        for ampa_id, ampa_group in group.groupby(ampa_col):
            aunp1_rows = ampa_group[ampa_group[type_col] == "AuNP_1"]
            aunp2_rows = ampa_group[ampa_group[type_col] == "AuNP_2"]

            if len(aunp1_rows) != 1 or len(aunp2_rows) != 1:
                logger.warning(
                    f"Skipping AMPA {ampa_id} in {tomo_name}: "
                    f"expected 1 AuNP_1 and 1 AuNP_2, got "
                    f"{len(aunp1_rows)} and {len(aunp2_rows)}"
                )
                continue

            aunp1 = aunp1_rows[coord_cols].values[0]
            aunp2 = aunp2_rows[coord_cols].values[0]
            ampas.append({
                "ampa_id": int(ampa_id),
                "aunp1": aunp1,
                "aunp2": aunp2,
            })

        tomo_ampas[tomo_name] = ampas

    logger.info(f"Loaded {sum(len(v) for v in tomo_ampas.values())} AMPAs "
                f"from {len(tomo_ampas)} tomograms")
    return tomo_ampas


def build_ampa_proximity_graph(ampas: list, max_distance: float):
    """Build AMPA-level proximity graph.

    Two AMPAs are connected if the minimum distance between any of their
    AuNPs is <= max_distance.

    Args:
        ampas: List of AMPA dicts with aunp1/aunp2 coordinates
        max_distance: Maximum AuNP-AuNP distance for connection

    Returns:
        List of (i, j) index pairs of connected AMPAs
    """
    n = len(ampas)
    if n < 2:
        return []

    # Collect all AuNP positions and build a mapping back to AMPA index
    all_positions = []
    aunp_to_ampa = []
    for i, ampa in enumerate(ampas):
        all_positions.append(ampa["aunp1"])
        all_positions.append(ampa["aunp2"])
        aunp_to_ampa.extend([i, i])

    all_positions = np.array(all_positions)
    aunp_to_ampa = np.array(aunp_to_ampa)

    # Find all AuNP pairs within max_distance
    kdt = cKDTree(all_positions)
    pairs = kdt.query_pairs(max_distance, output_type="ndarray")

    # Convert to AMPA-level edges (skip intra-AMPA pairs)
    ampa_edges = set()
    for a, b in pairs:
        ampa_a = aunp_to_ampa[a]
        ampa_b = aunp_to_ampa[b]
        if ampa_a != ampa_b:
            edge = (min(ampa_a, ampa_b), max(ampa_a, ampa_b))
            ampa_edges.add(edge)

    return list(ampa_edges)


def extract_ampa_pairs(ampas: list, edges: list):
    """Extract connected pairs of AMPAs from the AMPA graph.

    Uses connected component analysis to find pairs (size-2 components)
    and enumerates all pairs within larger components.

    Args:
        ampas: List of AMPA dicts
        edges: List of (i, j) AMPA-level edges

    Returns:
        List of (ampa_i, ampa_j) tuples, and component sizes for each pair
    """
    import rustworkx as rx

    n = len(ampas)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(n))
    for i, j in edges:
        graph.add_edge(i, j, None)

    components = rx.connected_components(graph)

    pairs = []
    comp_sizes = []

    for comp in components:
        comp = sorted(comp)
        if len(comp) < 2:
            continue

        # Extract all pairs of AMPAs that are directly connected
        for i, j in combinations(comp, 2):
            if (min(i, j), max(i, j)) in set(edges):
                pairs.append((i, j))
                comp_sizes.append(len(comp))

    return pairs, comp_sizes


def make_subgraphs_from_ampa_pairs(ampas, ampa_pairs):
    """Create 4-AuNP subgraphs from AMPA pairs with known ordering.

    Ordering: [AMPA_A.AuNP_1, AMPA_A.AuNP_2, AMPA_B.AuNP_1, AMPA_B.AuNP_2]
    This way positions [0,1] = pair 1, positions [2,3] = pair 2.

    Args:
        ampas: List of AMPA dicts
        ampa_pairs: List of (i, j) AMPA index pairs

    Returns:
        Tensor of shape (n_pairs, 4, 3)
    """
    subgraphs = []
    for i, j in ampa_pairs:
        coords = np.array([
            ampas[i]["aunp1"],
            ampas[i]["aunp2"],
            ampas[j]["aunp1"],
            ampas[j]["aunp2"],
        ])
        subgraphs.append(coords)

    if not subgraphs:
        return torch.empty(0, 4, 3, dtype=torch.float32)

    return torch.tensor(np.array(subgraphs), dtype=torch.float32)


def compute_features_known_pairing(subgraph_coords: torch.Tensor):
    """Compute pair-geometry features using known pairing (2D membrane projection).

    Positions [0,1] are pair 1 (AMPA_A), positions [2,3] are pair 2 (AMPA_B).
    No minimum-distance matching needed.

    Features:
        d_1, d_2: intra-pair distances (d_1 <= d_2)
        d_cc: centroid-centroid distance in XY
        theta_rot: signed angle between pair axes
        theta_cc: signed angle from pair-1 axis to centroid-centroid vector

    Args:
        subgraph_coords: Tensor of shape (n, 4, 3) in membrane-aligned frame

    Returns:
        (features: np.ndarray (n, 5), feature_names: list)
    """
    coords = subgraph_coords.numpy()
    n = coords.shape[0]
    xy = coords[:, :, :2]  # Project to membrane plane

    features = np.zeros((n, 5))

    for i in range(n):
        pts = xy[i]  # (4, 2)

        # Known pairing: pair 1 = (0, 1), pair 2 = (2, 3)
        a1, b1 = 0, 1
        a2, b2 = 2, 3

        d1 = np.linalg.norm(pts[b1] - pts[a1])
        d2 = np.linalg.norm(pts[b2] - pts[a2])

        # Order pairs by intra-pair distance (d1 <= d2) for canonical form
        if d1 > d2:
            (a1, b1), (a2, b2) = (a2, b2), (a1, b1)
            d1, d2 = d2, d1

        # Pair axis directions and centroids
        v1 = pts[b1] - pts[a1]
        v2 = pts[b2] - pts[a2]
        c1 = (pts[a1] + pts[b1]) / 2
        c2 = (pts[a2] + pts[b2]) / 2
        sep = c2 - c1
        d_inter = np.linalg.norm(sep)

        if d_inter < 1e-10 or np.linalg.norm(v1) < 1e-10:
            angle_between = 0.0
            angle_sep = 0.0
        else:
            v1c = _canonicalize_direction(v1, sep)
            v2c = _canonicalize_direction(v2, v1c)
            v1n = v1c / np.linalg.norm(v1c)
            sep_hat = sep / d_inter

            if np.linalg.norm(v2) < 1e-10:
                angle_between = 0.0
            else:
                v2n = v2c / np.linalg.norm(v2c)
                angle_between = _signed_angle_from_to(v1n, v2n)

            angle_sep = _signed_angle_from_to(v1n, sep_hat)

        features[i] = [d1, d2, d_inter, angle_between, angle_sep]

    feature_names = ["d_1", "d_2", "d_cc", "theta_rot", "theta_cc"]

    logger.info(
        f"Computed 5 pair-geometry features (known pairing, 2D) for {n} subgraphs "
        f"(d_1: {features[:, 0].mean():.1f} nm, d_2: {features[:, 1].mean():.1f} nm, "
        f"d_cc: {features[:, 2].mean():.1f} nm, "
        f"theta_rot: {np.degrees(features[:, 3]).mean():.1f} deg, "
        f"theta_cc: {np.degrees(features[:, 4]).mean():.1f} deg)"
    )

    return features, feature_names


def _canonicalize_direction_3d(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Orient 3D vector v so that dot(v, ref) >= 0."""
    if np.dot(v, ref) < 0:
        return -v
    return v.copy()


def _unsigned_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle between two 3D vectors, in [0, pi]."""
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_features_3d(subgraph_coords: torch.Tensor):
    """Compute pair-geometry features using known pairing in full 3D.

    No projection to any plane — uses full 3D coordinates.
    Angles are unsigned (coordinate-frame-independent).

    Positions [0,1] are pair 1 (AMPA_A), positions [2,3] are pair 2 (AMPA_B).

    Features:
        d_1, d_2: intra-pair distances in 3D (d_1 <= d_2)
        d_cc: centroid-centroid distance in 3D
        theta_rot: unsigned angle between pair axes [0, pi/2]
            (canonicalized so both axes point "with" the separation vector)
        theta_cc: unsigned angle from pair-1 axis to centroid-centroid vector [0, pi/2]

    Args:
        subgraph_coords: Tensor of shape (n, 4, 3)

    Returns:
        (features: np.ndarray (n, 5), feature_names: list)
    """
    coords = subgraph_coords.numpy()
    n = coords.shape[0]

    features = np.zeros((n, 5))

    for i in range(n):
        pts = coords[i]  # (4, 3)

        # Known pairing: pair 1 = (0, 1), pair 2 = (2, 3)
        a1, b1 = 0, 1
        a2, b2 = 2, 3

        d1 = np.linalg.norm(pts[b1] - pts[a1])
        d2 = np.linalg.norm(pts[b2] - pts[a2])

        # Order pairs by intra-pair distance (d1 <= d2) for canonical form
        if d1 > d2:
            (a1, b1), (a2, b2) = (a2, b2), (a1, b1)
            d1, d2 = d2, d1

        # Pair axis directions and centroids (full 3D)
        v1 = pts[b1] - pts[a1]
        v2 = pts[b2] - pts[a2]
        c1 = (pts[a1] + pts[b1]) / 2
        c2 = (pts[a2] + pts[b2]) / 2
        sep = c2 - c1
        d_inter = np.linalg.norm(sep)

        if d_inter < 1e-10 or np.linalg.norm(v1) < 1e-10:
            angle_between = 0.0
            angle_sep = 0.0
        else:
            # Canonicalize: flip axes so they point "with" the separation vector
            v1c = _canonicalize_direction_3d(v1, sep)
            v2c = _canonicalize_direction_3d(v2, v1c)
            v1n = v1c / np.linalg.norm(v1c)
            sep_hat = sep / d_inter

            if np.linalg.norm(v2) < 1e-10:
                angle_between = 0.0
            else:
                v2n = v2c / np.linalg.norm(v2c)
                angle_between = _unsigned_angle_3d(v1n, v2n)

            angle_sep = _unsigned_angle_3d(v1n, sep_hat)

        features[i] = [d1, d2, d_inter, angle_between, angle_sep]

    feature_names = ["d_1", "d_2", "d_cc", "theta_rot", "theta_cc"]

    logger.info(
        f"Computed 5 pair-geometry features (known pairing, 3D) for {n} subgraphs "
        f"(d_1: {features[:, 0].mean():.1f} nm, d_2: {features[:, 1].mean():.1f} nm, "
        f"d_cc: {features[:, 2].mean():.1f} nm, "
        f"theta_rot: {np.degrees(features[:, 3]).mean():.1f} deg, "
        f"theta_cc: {np.degrees(features[:, 4]).mean():.1f} deg)"
    )

    return features, feature_names


def plot_paired_diagnostics(
    ampas: list,
    edges: list,
    ampa_pairs: list,
    comp_sizes: list,
    tomo_name: str,
    output_path: Path,
    active_zonogram_path: Path = None,
    pair_id_offset: int = 0,
):
    """Create diagnostic plot for paired AMPA analysis.

    Shows all AuNPs with:
    - Thin lines connecting AuNPs within each AMPA (known pairs)
    - Dashed lines for inter-AMPA proximity edges
    - Connected components colored and labeled with size
    - AMPA centroids marked
    - Numbered labels for each AMPA pair (global ID = pair_id_offset + local index)
    """
    import matplotlib.pyplot as plt
    import mrcfile
    import rustworkx as rx

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nm_to_plot = None

    # Load active zonogram background if available
    if active_zonogram_path and active_zonogram_path.exists():
        with mrcfile.open(active_zonogram_path, permissive=True) as mrc:
            az_data_mrc = mrc.data.copy()
        metadata_path = active_zonogram_path.with_suffix(".npy")
        if metadata_path.exists():
            az_metadata = np.load(metadata_path, allow_pickle=True).tolist()
            az_center = az_metadata["center"]
            az_cs = az_metadata["cs"]
            az_offset = np.floor(np.array(az_data_mrc.shape)[[2, 1, 0]] / 2)

            def nm_to_plot(nm_coords):
                return (np.atleast_2d(nm_coords) - az_center) @ az_cs.T + az_offset

            az_projection = az_data_mrc.min(axis=0)
            vmin, vmax = np.percentile(az_projection, [1, 99])
            ax.imshow(az_projection, cmap="gray", interpolation="mitchell",
                      origin="lower", alpha=0.8, vmin=vmin, vmax=vmax)
            ax.set_xlim(0, az_projection.shape[1])
            ax.set_ylim(0, az_projection.shape[0])

    def _to_plot(coords_nm):
        if nm_to_plot is not None:
            return nm_to_plot(coords_nm)
        return np.atleast_2d(coords_nm)

    # Build AMPA-level rustworkx graph for component analysis
    n = len(ampas)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(n))
    for i, j in edges:
        graph.add_edge(i, j, None)
    components = rx.connected_components(graph)

    # Color map for components
    cmap = plt.cm.tab20
    node_to_comp_idx = {}
    comp_info = []
    for ci, comp in enumerate(components):
        for node in comp:
            node_to_comp_idx[node] = ci
        # Component centroid (average of AMPA centroids)
        centroids = []
        for node in comp:
            c = (np.array(ampas[node]["aunp1"]) + np.array(ampas[node]["aunp2"])) / 2
            centroids.append(c)
        center = np.mean(centroids, axis=0)
        comp_info.append({"center": center, "size": len(comp), "ci": ci})

    # Draw inter-AMPA edges (dashed)
    for i, j in edges:
        ci_a = (np.array(ampas[i]["aunp1"]) + np.array(ampas[i]["aunp2"])) / 2
        ci_b = (np.array(ampas[j]["aunp1"]) + np.array(ampas[j]["aunp2"])) / 2
        p1 = _to_plot(ci_a)[0]
        p2 = _to_plot(ci_b)[0]
        color = cmap(node_to_comp_idx[i] % 20)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                alpha=0.5, linewidth=1.5, linestyle="--", zorder=2)

    # Draw each AMPA: intra-pair line + AuNP dots + centroid
    for idx, ampa in enumerate(ampas):
        a1 = _to_plot(ampa["aunp1"])[0]
        a2 = _to_plot(ampa["aunp2"])[0]
        ci = node_to_comp_idx.get(idx, 0)
        color = cmap(ci % 20)

        # Intra-AMPA line (solid, thin)
        ax.plot([a1[0], a2[0]], [a1[1], a2[1]], color=color,
                alpha=0.8, linewidth=2.0, solid_capstyle="round", zorder=3)

        # AuNP dots
        ax.scatter([a1[0], a2[0]], [a1[1], a2[1]], c=[color], s=25,
                   edgecolors="white", linewidth=0.5, zorder=4)

        # Centroid (small diamond)
        centroid = (a1 + a2) / 2
        ax.scatter(centroid[0], centroid[1], c=[color], s=40, marker="D",
                   edgecolors="black", linewidth=0.5, alpha=0.7, zorder=5)

    # Label each AMPA pair with its global ID at the midpoint of the two AMPA centroids
    for local_idx, (i, j) in enumerate(ampa_pairs):
        ci_a = (np.array(ampas[i]["aunp1"]) + np.array(ampas[i]["aunp2"])) / 2
        ci_b = (np.array(ampas[j]["aunp1"]) + np.array(ampas[j]["aunp2"])) / 2
        mid = (ci_a + ci_b) / 2
        p = _to_plot(mid)[0]
        global_id = pair_id_offset + local_idx
        ci = node_to_comp_idx.get(i, 0)
        color = cmap(ci % 20)
        ax.text(p[0], p[1], str(global_id),
                fontsize=6, ha="center", va="center", color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                          edgecolor="white", alpha=0.85, linewidth=0.5),
                zorder=10)

    xlabel = "X (pixels)" if nm_to_plot is not None else "X (nm)"
    ylabel = "Y (pixels)" if nm_to_plot is not None else "Y (nm)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{tomo_name}\n"
        f"{len(ampas)} AMPAs, {len(edges)} inter-AMPA edges, "
        f"{len(ampa_pairs)} AMPA pairs",
        fontsize=14, fontweight="bold",
    )
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved diagnostic plot to {output_path}")


def run_paired_pipeline(config: AnalysisConfig, paired_star_path: Path,
                        force: bool = False, skip_membrane_alignment: bool = False,
                        use_3d_angles: bool = False):
    """Run the full paired analysis pipeline."""
    logger.info("=" * 60)
    logger.info("PAIRED AuNP ANALYSIS")
    logger.info("=" * 60)

    # Check for existing output
    features_path = config.get_precomputed_path("features.npy")
    if features_path.exists() and not force:
        logger.warning(f"Features already exist at {features_path}")
        logger.warning("Use --force to recompute")
        return False

    # Load paired star file
    logger.info(f"Loading paired AuNPs from {paired_star_path}")
    tomo_ampas = load_paired_star_file(paired_star_path)

    # Build tomo name -> config tomo mapping for membrane data
    tomo_to_config = {}
    for tomo in config.tomos:
        tomo_to_config[tomo[0]] = tomo

    # Process each tomogram
    subgraphs_list = []
    original_list = []
    comp_size_list = []
    tomo_names = []
    tomo_pack_info = []
    global_pair_offset = 0

    for tomo_name in sorted(tomo_ampas.keys()):
        if tomo_name not in tomo_to_config:
            logger.info(f"Skipping {tomo_name}: not in config tomogram list")
            continue

        ampas = tomo_ampas[tomo_name]
        if len(ampas) < 2:
            logger.info(f"Skipping {tomo_name}: only {len(ampas)} AMPAs")
            continue

        tomo_config = tomo_to_config[tomo_name]
        az_id = tomo_config[2] if len(tomo_config) == 3 else tomo_config[1]

        logger.info(f"\nProcessing {tomo_name} ({len(ampas)} AMPAs)")

        # Build AMPA proximity graph
        edges = build_ampa_proximity_graph(ampas, config.kdt_max_distance)
        logger.info(f"  AMPA graph: {len(ampas)} nodes, {len(edges)} edges")

        # Extract connected AMPA pairs
        ampa_pairs, comp_sizes = extract_ampa_pairs(ampas, edges)
        logger.info(f"  Found {len(ampa_pairs)} AMPA pairs "
                    f"(from components of sizes: {sorted(set(comp_sizes)) if comp_sizes else 'none'})")

        if not ampa_pairs:
            continue

        # Create 4-AuNP subgraphs with known ordering
        subgraphs = make_subgraphs_from_ampa_pairs(ampas, ampa_pairs)
        original_subgraphs = subgraphs.clone()

        # Align to membrane normal (unless disabled)
        if skip_membrane_alignment:
            logger.info("  Skipping membrane alignment (--no-membrane-alignment)")
        else:
            glb_path = config.get_membrane_glb_path(tomo_config)
            if glb_path.exists():
                mem_positions, mem_normals = load_membrane_mesh(glb_path)
                subgraphs = align_subgraphs_to_membrane(
                    subgraphs, mem_positions, mem_normals
                )
            else:
                logger.warning(f"  No membrane mesh at {glb_path}, skipping alignment")

        subgraphs_list.append(subgraphs)
        original_list.append(original_subgraphs)
        comp_size_list.append(np.array(comp_sizes))
        tomo_names.append(f"{tomo_name}_{az_id}")
        tomo_pack_info.append(len(ampa_pairs))

        # Generate diagnostic plot for this tomogram
        if config.generate_diagnostics:
            az_path = config.get_active_zonogram_path(tomo_config)
            diag_path = config.diagnostics_folder / f"{tomo_name}_{az_id}_paired.png"
            plot_paired_diagnostics(
                ampas=ampas,
                edges=edges,
                ampa_pairs=ampa_pairs,
                comp_sizes=comp_sizes,
                tomo_name=f"{tomo_name}_{az_id}",
                output_path=diag_path,
                active_zonogram_path=az_path,
                pair_id_offset=global_pair_offset,
            )

        global_pair_offset += len(ampa_pairs)

    if not subgraphs_list:
        logger.error("No subgraphs found!")
        return False

    # Add decoy subgraphs if configured
    extra_labels = []
    if config.decoy_star_file:
        from graph_construction import generate_decoy_subgraphs
        decoys = generate_decoy_subgraphs(config)
        if len(decoys) > 0:
            subgraphs_list.append(decoys)
            original_list.append(decoys)
            comp_size_list.append(np.zeros(len(decoys), dtype=int))
            tomo_pack_info.append(len(decoys))
            extra_labels.append("Liquid Decoy")
            logger.info(f"Added {len(decoys)} decoy subgraphs")

    # Pack all subgraphs
    packed_subgraphs, pack_info = einops.pack(subgraphs_list, pattern="* p d")
    packed_original, _ = einops.pack(original_list, pattern="* p d")
    packed_comp_sizes = np.concatenate(comp_size_list)

    logger.info(f"\nTotal subgraphs: {len(packed_subgraphs)}")
    logger.info(f"Per tomogram: {list(zip(tomo_names + extra_labels, tomo_pack_info))}")

    # Compute features with known pairing
    if use_3d_angles:
        logger.info("\nComputing pair-geometry features (known pairing, 3D)...")
        features, feature_names = compute_features_3d(packed_subgraphs)
    else:
        logger.info("\nComputing pair-geometry features (known pairing, 2D)...")
        features, feature_names = compute_features_known_pairing(packed_subgraphs)

    # Save everything
    save_precomputed_data(
        config.get_precomputed_path("subgraphs.pt"), packed_subgraphs
    )
    save_precomputed_data(
        config.get_precomputed_path("subgraphs_original.pt"), packed_original
    )
    np.save(config.get_precomputed_path("features.npy"), features)

    with open(config.get_precomputed_path("feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save metadata (compatible with existing plot scripts)
    metadata = {
        "tomos": [list(tomo_to_config[tn.rsplit("_", 1)[0]])
                  for tn in tomo_names if tn.rsplit("_", 1)[0] in tomo_to_config],
        "tomo_names": tomo_names,
        "tomo_pack_info": tomo_pack_info,
        "kdt_max_distance": config.kdt_max_distance,
        "kdt_min_distance": config.kdt_min_distance,
        "subgraph_size": config.subgraph_size,
        "include_test_shape": config.include_test_shape,
        "component_sizes": [int(s) for s in packed_comp_sizes],
        "analysis_type": "paired",
        "angle_mode": "3d" if use_3d_angles else "2d",
        "paired_star_file": str(paired_star_path),
    }
    with open(config.get_precomputed_path("metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate summary figure
    if config.generate_diagnostics:
        from visualization import create_summary_figure
        summary_info = [
            {"name": tn, "n_points": tp * 4, "n_subgraphs": tp}
            for tn, tp in zip(tomo_names, tomo_pack_info)
            if tn not in ("Liquid Decoy", "Test Shape")
        ]
        summary_path = config.diagnostics_folder / "summary.png"
        create_summary_figure(summary_info, summary_path)
        logger.info(f"Diagnostic plots saved to: {config.diagnostics_folder}")

    logger.info("\n" + "=" * 60)
    logger.info("Paired analysis complete!")
    logger.info(f"Results saved to: {config.output_folder}")
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Precompute paired AuNP analysis",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config_paired.toml"),
        help="Path to configuration file (default: config_paired.toml)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recomputation even if files exist",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--no-membrane-alignment", action="store_true",
        help="Skip membrane normal alignment (project in raw tomogram frame)",
    )
    parser.add_argument(
        "--3d-angles", action="store_true", dest="use_3d_angles",
        help="Compute angles in full 3D (no projection). Implies --no-membrane-alignment.",
    )
    args = parser.parse_args()

    # 3D angles implies no membrane alignment (no plane to project onto)
    if args.use_3d_angles:
        args.no_membrane_alignment = True

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load raw TOML to extract paired_star_file before passing to AnalysisConfig
    import tomllib
    with open(args.config, "rb") as f:
        raw = tomllib.load(f)
    analysis_dict = raw.get("analysis", raw)
    paired_star_path = Path(analysis_dict.pop(
        "paired_star_file",
        "inputs/all_ampa_poses_ilp_aunp5.0-9.0nm_mem15.0-22.0nm_steric5.0nm_paired_aunps.star",
    ))

    # Reconstruct config TOML without the paired_star_file key
    config = AnalysisConfig.from_toml_or_default(args.config)
    # Workaround: if from_toml failed due to unknown key, build manually
    if not config.tomos:
        config = AnalysisConfig(**{
            k: v for k, v in analysis_dict.items()
            if k in AnalysisConfig.__dataclass_fields__
        })

    if not paired_star_path.exists():
        logger.error(f"Paired star file not found: {paired_star_path}")
        sys.exit(1)

    run_paired_pipeline(config, paired_star_path, force=args.force,
                        skip_membrane_alignment=args.no_membrane_alignment,
                        use_3d_angles=args.use_3d_angles)


if __name__ == "__main__":
    main()
