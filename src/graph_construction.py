"""Graph construction utilities for finding connected subgraphs."""

import numpy as np
import torch
import rustworkx as rx
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def build_proximity_graph(
    coordinates: np.ndarray,
    max_distance: float,
    min_distance: float = 0.0
) -> Tuple[rx.PyGraph, np.ndarray]:
    """
    Build a graph where nodes are connected if they are within a distance range.

    Args:
        coordinates: Array of shape (n_points, 3) with XYZ coordinates
        max_distance: Maximum distance for edge creation
        min_distance: Minimum distance for edge creation (default: 0.0)

    Returns:
        Tuple of:
            - rustworkx graph with nodes for each coordinate and edges for nearby pairs
            - Array of connected component sizes
    """
    # Build KD-tree for efficient neighbor search
    kdt = cKDTree(coordinates)

    # Find all pairs within max_distance
    pairs = kdt.query_pairs(max_distance, output_type="ndarray")

    # Filter by minimum distance
    if min_distance > 0:
        distances = np.linalg.norm(
            coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]], axis=1
        )
        pairs = pairs[distances > min_distance]

    # Build graph
    graph = rx.PyGraph()
    graph.add_nodes_from(range(len(coordinates)))

    for i, j in pairs:
        graph.add_edge(i, j, None)

    # Get connected component sizes
    connected_components = rx.connected_components(graph)
    component_sizes = np.array([len(comp) for comp in connected_components])

    logger.info(
        f"Built graph with {len(coordinates)} nodes and {len(pairs)} edges "
        f"(distance range: {min_distance}-{max_distance})"
    )
    logger.info(
        f"Found {len(connected_components)} connected components, "
        f"sizes: {np.min(component_sizes)}-{np.max(component_sizes)}"
    )

    return graph, component_sizes


def extract_connected_subgraphs(
    graph: rx.PyGraph,
    subgraph_size: int,
    coordinates: np.ndarray
) -> torch.Tensor:
    """
    Extract all connected subgraphs of a specific size.

    Args:
        graph: rustworkx graph
        subgraph_size: Number of nodes in each subgraph
        coordinates: Original coordinates array (n_points, 3)

    Returns:
        Tensor of shape (n_subgraphs, subgraph_size, 3) with subgraph coordinates
    """
    # Find connected subgraphs
    subgraph_indices = rx.connected_subgraphs(graph, subgraph_size)

    if len(subgraph_indices) == 0:
        logger.warning(f"No connected subgraphs of size {subgraph_size} found")
        return torch.empty(0, subgraph_size, 3)

    # Extract coordinates for each subgraph
    subgraph_coords = coordinates[subgraph_indices]

    logger.info(f"Found {len(subgraph_indices)} connected subgraphs of size {subgraph_size}")

    return torch.tensor(subgraph_coords, dtype=torch.float32)


def process_tomogram_graphs(
    coordinate_list: List[torch.Tensor],
    config,
    return_graphs: bool = False
) -> Tuple[torch.Tensor, List[Tuple[int]], Optional[List[dict]]]:
    """
    Process all tomograms to extract connected subgraphs.

    Args:
        coordinate_list: List of coordinate tensors, one per tomogram
        config: AnalysisConfig instance with graph parameters
        return_graphs: If True, return graph objects and metadata for visualization

    Returns:
        Tuple of:
            - Packed subgraph tensor (n_total_subgraphs, subgraph_size, 3)
            - List of tuples indicating how many subgraphs came from each tomogram
            - (Optional) List of dicts with graph metadata for each tomogram
    """
    import einops

    subgraphs_tensors = []
    graph_metadata = [] if return_graphs else None

    for i, coords in enumerate(coordinate_list):
        coords_np = coords.numpy()

        # Build proximity graph
        graph, component_sizes = build_proximity_graph(
            coords_np,
            max_distance=config.kdt_max_distance,
            min_distance=config.kdt_min_distance
        )

        # Extract subgraphs
        subgraphs = extract_connected_subgraphs(
            graph,
            config.subgraph_size,
            coords_np
        )

        subgraphs_tensors.append(subgraphs)

        # Store graph metadata if requested
        if return_graphs:
            graph_metadata.append({
                'coordinates': coords_np,
                'graph': graph,
                'subgraphs': subgraphs,
                'n_points': len(coords_np),
                'n_subgraphs': len(subgraphs),
                'component_sizes': component_sizes
            })

    # Add test shape if configured
    if config.include_test_shape and config.test_shape_coordinates:
        test_shape = torch.tensor(
            [config.test_shape_coordinates],
            dtype=torch.float32
        )
        subgraphs_tensors.append(test_shape)
        logger.info("Added test shape to subgraphs")

    # Pack all subgraphs into a single tensor
    packed_tensor, pack_info = einops.pack(subgraphs_tensors, pattern="* p d")

    logger.info(f"Packed {len(subgraphs_tensors)} tomogram subgraphs into tensor of shape {packed_tensor.shape}")
    logger.info(f"Subgraphs per tomogram: {pack_info}")

    return packed_tensor, pack_info, graph_metadata
