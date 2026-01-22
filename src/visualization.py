"""Visualization utilities for diagnostic plots."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import mrcfile

logger = logging.getLogger(__name__)


def plot_graph_diagnostics(
    coordinates: np.ndarray,
    graph,
    subgraphs: torch.Tensor,
    tomogram_name: str,
    output_path: Path,
    active_zonogram_path: Optional[Path] = None,
    n_highlight: int = 5,
    figsize: Tuple[int, int] = (12, 10),
    plot_type: str = "both"
):
    """
    Create diagnostic plot showing the active zone with graph and highlighted subgraphs.

    Args:
        coordinates: All AUNP coordinates (n_points, 3) in global coordinates
        graph: rustworkx graph object
        subgraphs: Extracted subgraph coordinates (n_subgraphs, subgraph_size, 3)
        tomogram_name: Name of the tomogram for the title
        output_path: Path to save the figure
        active_zonogram_path: Path to the active zonogram MRC file (optional)
        n_highlight: Number of subgraphs to highlight in color
        figsize: Figure size
        plot_type: "subgraphs", "component_size", or "both" (default: "both")
    """
    if plot_type == "both":
        # Create both versions
        _plot_single_diagnostic(coordinates, graph, subgraphs, tomogram_name,
                               output_path.parent / (output_path.stem + "_subgraphs.png"),
                               active_zonogram_path, n_highlight, figsize, "subgraphs")
        _plot_single_diagnostic(coordinates, graph, subgraphs, tomogram_name,
                               output_path.parent / (output_path.stem + "_component_size.png"),
                               active_zonogram_path, n_highlight, figsize, "component_size")
    else:
        _plot_single_diagnostic(coordinates, graph, subgraphs, tomogram_name,
                               output_path, active_zonogram_path, n_highlight, figsize, plot_type)


def _plot_single_diagnostic(
    coordinates: np.ndarray,
    graph,
    subgraphs: torch.Tensor,
    tomogram_name: str,
    output_path: Path,
    active_zonogram_path: Optional[Path],
    n_highlight: int,
    figsize: Tuple[int, int],
    plot_type: str
):
    """
    Create a single diagnostic plot.

    Args:
        coordinates: All AUNP coordinates (n_points, 3) in global coordinates
        graph: rustworkx graph object
        subgraphs: Extracted subgraph coordinates
        tomogram_name: Name of the tomogram
        output_path: Path to save the figure
        active_zonogram_path: Path to active zonogram MRC
        n_highlight: Number of subgraphs to highlight
        figsize: Figure size
        plot_type: "subgraphs" or "component_size"
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Load and display active zonogram if available
    if active_zonogram_path and active_zonogram_path.exists():
        # Load active zonogram
        with mrcfile.open(active_zonogram_path, permissive=True) as mrc:
            az_data_mrc = torch.tensor(mrc.data.copy())
            az_shape = az_data_mrc.shape

        # Load metadata (center and coordinate system)
        metadata_path = active_zonogram_path.with_suffix(".npy")
        if metadata_path.exists():
            az_metadata = np.load(metadata_path, allow_pickle=True).tolist()
            center = az_metadata["center"]
            cs = az_metadata["cs"]

            # Transform coordinates to active zone coordinate system
            coords_transformed = (coordinates - center) @ cs.T
            coords_transformed += np.floor(np.array(az_shape)[[2, 1, 0]] / 2)

            # Display active zone projection (min along z-axis)
            az_projection = az_data_mrc.min(dim=0)[0].numpy()

            # Apply contrast stretching (percentile-based)
            vmin, vmax = np.percentile(az_projection, [1, 99])
            ax.imshow(az_projection, cmap="gray", interpolation='mitchell',
                     origin="lower", alpha=0.8, vmin=vmin, vmax=vmax)

            # Set extent to match the projection shape
            ax.set_xlim(0, az_projection.shape[1])
            ax.set_ylim(0, az_projection.shape[0])

            # Use transformed coordinates for plotting
            plot_coords = coords_transformed

            # Transform subgraphs as well
            transformed_subgraphs = []
            for sg in subgraphs:
                sg_np = sg.numpy()
                sg_transformed = (sg_np - center) @ cs.T
                sg_transformed += np.floor(np.array(az_shape)[[2, 1, 0]] / 2)
                transformed_subgraphs.append(sg_transformed)

            xlabel, ylabel = 'X (pixels)', 'Y (pixels)'
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            plot_coords = coordinates
            transformed_subgraphs = [sg.numpy() for sg in subgraphs]
            xlabel, ylabel = 'X (nm)', 'Y (nm)'
    else:
        plot_coords = coordinates
        transformed_subgraphs = [sg.numpy() for sg in subgraphs]
        xlabel, ylabel = 'X (nm)', 'Y (nm)'

    # Get connected components if needed
    import rustworkx as rx
    connected_components = None

    if plot_type == "component_size":
        connected_components = rx.connected_components(graph)

        # Create color map for components
        n_components = len(connected_components)
        # Use a colormap with many distinct colors
        component_colors = plt.cm.tab20(np.linspace(0, 1, min(n_components, 20)))
        if n_components > 20:
            # Repeat colors if we have more than 20 components
            component_colors = np.tile(component_colors, (n_components // 20 + 1, 1))[:n_components]

        # Create mapping from node index to component color
        node_to_color = {}
        component_info = []

        for comp_idx, component in enumerate(connected_components):
            color = component_colors[comp_idx % len(component_colors)]
            for node in component:
                node_to_color[node] = color

            # Calculate component center for label
            component_coords = plot_coords[list(component)]
            center = np.mean(component_coords, axis=0)
            component_info.append({
                'center': center,
                'size': len(component),
                'color': color
            })

        # Plot edges colored by component
        edge_list = graph.edge_list()
        for i, j in edge_list:
            p1, p2 = plot_coords[i], plot_coords[j]
            color = node_to_color.get(i, [0.5, 0.5, 0.5, 1.0])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                   alpha=0.6, linewidth=1.0, zorder=1)

        # Plot points colored by component
        point_colors = [node_to_color.get(i, [0.5, 0.5, 0.5, 1.0]) for i in range(len(plot_coords))]
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1], c=point_colors, s=20,
                  alpha=0.7, edgecolors='white', linewidth=0.5, zorder=2)

        # Add text labels showing component sizes
        for info in component_info:
            ax.text(info['center'][0], info['center'][1], str(info['size']),
                   fontsize=8, ha='center', va='center',
                   color=info['color'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=info['color'], alpha=0.8),
                   zorder=10)

    else:  # subgraphs
        # Plot all points
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1], c='white', s=15,
                   alpha=0.6, label='All AUNPs', edgecolors='darkgray', linewidth=0.3)

        # Plot all edges in blue (changed from gray)
        edge_list = graph.edge_list()
        for i, j in edge_list:
            p1, p2 = plot_coords[i], plot_coords[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='steelblue',
                    alpha=0.5, linewidth=1.0, zorder=1)

        # Highlight selected subgraphs in color
        n_subgraphs = min(n_highlight, len(transformed_subgraphs))
        colors = plt.cm.tab10(np.linspace(0, 1, n_subgraphs))

        # Select subgraphs to highlight (evenly spaced)
        indices = np.linspace(0, len(transformed_subgraphs) - 1, n_subgraphs, dtype=int)

        # Add jitter to avoid exact overlap
        np.random.seed(42)  # For reproducibility

        for idx, color in zip(indices, colors):
            subgraph = transformed_subgraphs[idx]

            # Small jitter to make overlapping points visible
            jitter = np.random.normal(0, 0.5, size=2)

            # Plot points with more transparency
            ax.scatter(subgraph[:, 0] + jitter[0], subgraph[:, 1] + jitter[1],
                      c=[color], s=100, edgecolors='black', linewidth=2,
                      zorder=10, alpha=0.4)

            # Plot connections within subgraph (more transparent)
            for i in range(len(subgraph)):
                for j in range(i + 1, len(subgraph)):
                    p1, p2 = subgraph[i], subgraph[j]
                    ax.plot([p1[0] + jitter[0], p2[0] + jitter[0]],
                           [p1[1] + jitter[1], p2[1] + jitter[1]],
                           c=color, alpha=0.5, linewidth=3, zorder=9)

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if plot_type == "component_size" and connected_components is not None:
        title_extra = f'{len(connected_components)} components'
    else:
        title_extra = f'{len(subgraphs)} subgraphs'

    ax.set_title(f'{tomogram_name}\n{len(coordinates)} AUNPs, {title_extra}, '
                f'{len(graph.edge_list())} edges', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    if plot_type == "subgraphs":
        ax.legend(loc='upper right', framealpha=0.8)

    plt.tight_layout(pad=0.5)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved diagnostic plot to {output_path}")


def create_summary_figure(
    all_tomo_info: List[dict],
    output_path: Path,
    figsize: Tuple[int, int] = (16, 4)
):
    """
    Create a summary figure showing statistics for all tomograms.

    Args:
        all_tomo_info: List of dicts with keys: name, n_points, n_subgraphs
        output_path: Path to save the figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    names = [info['name'] for info in all_tomo_info]
    n_points = [info['n_points'] for info in all_tomo_info]
    n_subgraphs = [info['n_subgraphs'] for info in all_tomo_info]

    x = np.arange(len(names))
    width = 0.35

    # Bar plot for number of points
    ax1.bar(x, n_points, width, label='AUNPs', color='steelblue')
    ax1.set_xlabel('Tomogram')
    ax1.set_ylabel('Count')
    ax1.set_title('Number of AUNPs per Tomogram')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Bar plot for number of subgraphs
    ax2.bar(x, n_subgraphs, width, label='Subgraphs', color='coral')
    ax2.set_xlabel('Tomogram')
    ax2.set_ylabel('Count')
    ax2.set_title('Number of Subgraphs per Tomogram')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved summary figure to {output_path}")


def create_component_size_distribution(
    all_tomo_info: List[dict],
    output_path: Path,
    max_size: int = 20,
    figsize: Tuple[int, int] = (16, 10),
    count_type: str = "both"
):
    """
    Create a figure showing the distribution of connected component sizes.

    Args:
        all_tomo_info: List of dicts with keys: name, component_sizes
        output_path: Path to save the figure
        max_size: Maximum component size to show in detail (larger grouped)
        figsize: Figure size
        count_type: "subgraphs", "aunps", or "both" (default: "both")
    """
    if count_type == "both":
        # Create both versions
        _create_single_distribution(all_tomo_info, output_path.parent / (output_path.stem + "_subgraphs.png"),
                                   max_size, figsize, "subgraphs")
        _create_single_distribution(all_tomo_info, output_path.parent / (output_path.stem + "_aunps.png"),
                                   max_size, figsize, "aunps")
    else:
        _create_single_distribution(all_tomo_info, output_path, max_size, figsize, count_type)


def _create_single_distribution(
    all_tomo_info: List[dict],
    output_path: Path,
    max_size: int,
    figsize: Tuple[int, int],
    count_type: str
):
    """
    Create a single distribution figure.

    Args:
        all_tomo_info: List of dicts with keys: name, component_sizes
        output_path: Path to save the figure
        max_size: Maximum component size to show in detail
        figsize: Figure size
        count_type: "subgraphs" or "aunps"
    """
    n_tomos = len(all_tomo_info)

    # Create a grid of subplots
    n_cols = 2
    n_rows = (n_tomos + 1) // n_cols + 1  # +1 for combined plot

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Plot individual tomogram distributions
    all_component_sizes = []
    for i, info in enumerate(all_tomo_info):
        ax = axes[i]
        component_sizes = info['component_sizes']
        all_component_sizes.extend(component_sizes)

        # Create histogram
        bins = np.arange(1, max_size + 2) - 0.5
        counts, _ = np.histogram(component_sizes, bins=bins)

        x_pos = np.arange(1, max_size + 1)

        if count_type == "aunps":
            # Multiply by component size to get total AUNPs
            y_values = counts * x_pos
            ylabel = 'Count of AUNPs'

            # Add "larger" category
            larger_components = component_sizes[component_sizes >= max_size + 1]
            n_larger = np.sum(larger_components) if len(larger_components) > 0 else 0
        else:  # subgraphs
            y_values = counts
            ylabel = 'Count of Subgraphs'

            # Add "larger" category
            n_larger = np.sum(component_sizes >= max_size + 1)

        ax.bar(x_pos, y_values, color='steelblue', alpha=0.7, edgecolor='black')

        if n_larger > 0:
            ax.bar([max_size + 1], [n_larger], color='coral', alpha=0.7,
                  edgecolor='black', label=f'≥{max_size + 1}')
            ax.legend(fontsize=8)

        ax.set_xlabel('Component Size', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        total_aunps = np.sum(component_sizes)
        ax.set_title(f"{info['name']}\n({len(component_sizes)} components, {total_aunps} AUNPs)",
                    fontsize=10, fontweight='bold')
        ax.set_xticks(np.arange(1, max_size + 2, 2))
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0.5, max_size + 1.5)

    # Combined distribution
    ax_combined = axes[-2]
    all_component_sizes_array = np.array(all_component_sizes)

    bins = np.arange(1, max_size + 2) - 0.5
    counts, _ = np.histogram(all_component_sizes_array, bins=bins)

    x_pos = np.arange(1, max_size + 1)

    if count_type == "aunps":
        # Multiply by component size to get total AUNPs
        y_values_combined = counts * x_pos
        ylabel_combined = 'Count of AUNPs'

        # Add "larger" category
        larger_components_combined = all_component_sizes_array[all_component_sizes_array >= max_size + 1]
        n_larger_combined = np.sum(larger_components_combined) if len(larger_components_combined) > 0 else 0
    else:  # subgraphs
        y_values_combined = counts
        ylabel_combined = 'Count of Subgraphs'

        # Add "larger" category
        n_larger_combined = np.sum(all_component_sizes_array >= max_size + 1)

    ax_combined.bar(x_pos, y_values_combined, color='darkgreen', alpha=0.7, edgecolor='black')

    if n_larger_combined > 0:
        ax_combined.bar([max_size + 1], [n_larger_combined], color='coral', alpha=0.7,
                       edgecolor='black', label=f'≥{max_size + 1}')
        ax_combined.legend(fontsize=9)

    ax_combined.set_xlabel('Component Size', fontsize=10)
    ax_combined.set_ylabel(ylabel_combined, fontsize=10)
    total_components = len(all_component_sizes)
    total_aunps_combined = np.sum(all_component_sizes_array)
    ax_combined.set_title(f'Combined Distribution\n({total_components} components, {total_aunps_combined} AUNPs)',
                         fontsize=11, fontweight='bold')
    ax_combined.set_xticks(np.arange(1, max_size + 2, 2))
    ax_combined.grid(axis='y', alpha=0.3)
    ax_combined.set_xlim(0.5, max_size + 1.5)

    # Hide unused subplots
    for i in range(n_tomos + 1, len(axes)):
        axes[i].axis('off')

    title_suffix = "(Count of AUNPs)" if count_type == "aunps" else "(Count of Subgraphs)"
    plt.suptitle(f'Connected Component Size Distribution {title_suffix}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved component size distribution to {output_path}")
