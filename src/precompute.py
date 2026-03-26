#!/usr/bin/env python3
"""
Precompute expensive operations for graph pattern analysis.

This script supports multiple stages:
1. subgraphs: Load AUNP coordinates, build graphs, extract subgraphs
2. rmsd: Compute RMSD matrix (most expensive operation)
3. all: Run all stages

You can run stages independently:
  - First run: --stage subgraphs (generates diagnostics)
  - Then run: --stage rmsd (uses precomputed subgraphs)

Run this before using the interactive marimo app.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import AnalysisConfig
from data_loading import (
    load_all_coordinates,
    load_membrane_mesh,
    save_precomputed_data,
    load_precomputed_data,
)
from graph_construction import process_tomogram_graphs
from alignment import compute_rmsd_matrix, center_coordinates
from visualization import plot_graph_diagnostics, create_summary_figure, create_component_size_distribution


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_metadata(config: AnalysisConfig, tomo_pack_info: list, tomo_names: list, comp_sizes=None):
    """Save metadata about the computation."""
    metadata = {
        "tomos": config.tomos,
        "tomo_names": tomo_names,
        "kdt_max_distance": config.kdt_max_distance,
        "kdt_min_distance": config.kdt_min_distance,
        "subgraph_size": config.subgraph_size,
        "tomo_pack_info": [list(t) for t in tomo_pack_info],
        "include_test_shape": config.include_test_shape,
    }
    if comp_sizes is not None:
        metadata["component_sizes"] = [int(s) for s in comp_sizes]

    metadata_path = config.get_precomputed_path("metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved metadata to {metadata_path}")


def run_subgraph_stage(config: AnalysisConfig, force: bool = False) -> bool:
    """
    Stage 1: Load coordinates and extract subgraphs.

    Args:
        config: Analysis configuration
        force: Force recomputation even if files exist

    Returns:
        True if successful
    """
    logger = logging.getLogger(__name__)

    # Check if already computed
    subgraphs_path = config.get_precomputed_path("subgraphs.pt")
    if subgraphs_path.exists() and not force:
        logger.warning(f"Subgraphs already computed at {subgraphs_path}")
        logger.warning("Use --force to recompute")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Loading coordinates and extracting subgraphs")
    logger.info("=" * 60)

    # Load coordinates
    coordinate_list, tomo_names = load_all_coordinates(config)
    logger.info(f"Loaded coordinates from {len(coordinate_list)} tomograms")

    # Load membrane meshes for normal-constrained alignment
    membrane_data = []
    for tomo in config.tomos:
        glb_path = config.get_membrane_glb_path(tomo)
        logger.info(f"Loading membrane mesh from {glb_path}")
        positions, normals = load_membrane_mesh(glb_path)
        membrane_data.append((positions, normals))

    # Build graphs and extract subgraphs
    logger.info("Building graphs and extracting subgraphs...")
    subgraphs_tensor, tomo_pack_info, graph_metadata, original_subgraphs, comp_sizes = process_tomogram_graphs(
        coordinate_list, config,
        return_graphs=config.generate_diagnostics,
        membrane_data=membrane_data,
    )

    # Save subgraphs (membrane-aligned for RMSD computation)
    save_precomputed_data(subgraphs_path, subgraphs_tensor)

    # Save original (tomogram-frame) subgraphs for overlay visualization
    save_precomputed_data(
        config.get_precomputed_path("subgraphs_original.pt"),
        original_subgraphs
    )

    # Save metadata (include component sizes as list for JSON serialization)
    save_metadata(config, tomo_pack_info, tomo_names, comp_sizes)

    # Generate diagnostic plots
    if config.generate_diagnostics and graph_metadata:
        logger.info("\n" + "=" * 60)
        logger.info("Generating diagnostic plots")
        logger.info("=" * 60)

        summary_info = []
        component_info = []

        for i, (tomo, tomo_name, meta) in enumerate(zip(config.tomos, tomo_names, graph_metadata)):
            logger.info(f"Creating diagnostic plot for {tomo_name}...")

            # Get active zonogram path
            az_path = config.get_active_zonogram_path(tomo)

            output_path = config.diagnostics_folder / f"{tomo_name}_graph.png"
            plot_graph_diagnostics(
                coordinates=meta['coordinates'],
                graph=meta['graph'],
                subgraphs=meta['subgraphs'],
                tomogram_name=tomo_name,
                output_path=output_path,
                active_zonogram_path=az_path,
                n_highlight=config.n_subgraphs_to_highlight
            )

            summary_info.append({
                'name': tomo_name,
                'n_points': meta['n_points'],
                'n_subgraphs': meta['n_subgraphs']
            })

            component_info.append({
                'name': tomo_name,
                'component_sizes': meta['component_sizes']
            })

        # Create summary figures
        summary_path = config.diagnostics_folder / "summary.png"
        create_summary_figure(summary_info, summary_path)

        logger.info("Creating component size distribution plot...")
        component_dist_path = config.diagnostics_folder / "component_size_distribution.png"
        create_component_size_distribution(component_info, component_dist_path)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Subgraph extraction complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {config.output_folder}")
    if config.generate_diagnostics:
        logger.info(f"Diagnostic plots saved to: {config.diagnostics_folder}")

    return True


def run_rmsd_stage(config: AnalysisConfig, force: bool = False) -> bool:
    """
    Stage 2: Compute RMSD matrix.

    Args:
        config: Analysis configuration
        force: Force recomputation even if files exist

    Returns:
        True if successful
    """
    logger = logging.getLogger(__name__)

    # Check if RMSD already computed
    rmsd_path = config.get_precomputed_path("rmsd_matrix.pt")
    if rmsd_path.exists() and not force:
        logger.warning(f"RMSD matrix already computed at {rmsd_path}")
        logger.warning("Use --force to recompute")
        return False

    # Check if subgraphs exist
    subgraphs_path = config.get_precomputed_path("subgraphs.pt")
    if not subgraphs_path.exists():
        logger.error(f"Subgraphs not found at {subgraphs_path}")
        logger.error("Run with --stage subgraphs first")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Computing RMSD matrix")
    logger.info("=" * 60)
    logger.info("⚠️  This is computationally expensive and may take several minutes")

    # Load subgraphs
    subgraphs_tensor = load_precomputed_data(subgraphs_path)

    # Compute RMSD matrix (chunked to limit memory)
    best_rmsd, best_indices = compute_rmsd_matrix(subgraphs_tensor)

    # Save results
    save_precomputed_data(rmsd_path, best_rmsd)
    save_precomputed_data(
        config.get_precomputed_path("best_indices.pt"),
        best_indices
    )

    # Save centered coordinates
    centered = center_coordinates(subgraphs_tensor)
    save_precomputed_data(
        config.get_precomputed_path("centered_coords.pt"),
        centered
    )

    logger.info("\n" + "=" * 60)
    logger.info("✓ RMSD computation complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {config.output_folder}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Precompute graph pattern analysis data in stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run only subgraph extraction (with diagnostics)
  python src/precompute.py --stage subgraphs

  # Run only RMSD computation (requires subgraphs)
  python src/precompute.py --stage rmsd

  # Run all stages
  python src/precompute.py --stage all

  # Use custom config
  python src/precompute.py --config my_config.toml --stage all
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to configuration file (default: config.toml)"
    )
    parser.add_argument(
        "--stage",
        choices=["subgraphs", "rmsd", "all"],
        default="all",
        help="Which stage to run (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for precomputed data (overrides config file)"
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip generating diagnostic plots"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if files exist"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = AnalysisConfig.from_toml_or_default(args.config)

    # Override options from command line
    if args.output_dir:
        config.output_folder = args.output_dir
        config.output_folder.mkdir(parents=True, exist_ok=True)

    if args.no_diagnostics:
        config.generate_diagnostics = False

    logger.info("=" * 60)
    logger.info("AUNP Graph Pattern Analysis - Precomputation")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {config.output_folder}")
    logger.info(f"Diagnostics: {'enabled' if config.generate_diagnostics else 'disabled'}")
    logger.info(f"Processing {len(config.tomos)} tomograms")
    logger.info(f"Stage: {args.stage}")

    # Run requested stage(s)
    success = True

    if args.stage in ["subgraphs", "all"]:
        success = run_subgraph_stage(config, args.force)
        if not success and not args.force:
            logger.info("Skipping subgraph stage (already completed)")
            success = True

    if args.stage in ["rmsd", "all"]:
        if success or args.stage == "rmsd":
            success = run_rmsd_stage(config, args.force)
            if not success and not args.force:
                logger.info("Skipping RMSD stage (already completed)")
                success = True

    # Final message
    if success or not args.force:
        logger.info("\n" + "=" * 60)
        logger.info("You can now run the interactive marimo app:")
        logger.info("  marimo run src/marimo_embedding_app.py")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
