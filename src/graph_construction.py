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
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Extract all connected subgraphs of a specific size.

    Args:
        graph: rustworkx graph
        subgraph_size: Number of nodes in each subgraph
        coordinates: Original coordinates array (n_points, 3)

    Returns:
        Tuple of:
            - Tensor of shape (n_subgraphs, subgraph_size, 3) with subgraph coordinates
            - Array of component sizes that each subgraph belongs to
    """
    # Find connected subgraphs
    subgraph_indices = rx.connected_subgraphs(graph, subgraph_size)

    if len(subgraph_indices) == 0:
        logger.warning(f"No connected subgraphs of size {subgraph_size} found")
        return torch.empty(0, subgraph_size, 3), np.array([], dtype=int)

    # Build node-to-component-size mapping
    connected_components = rx.connected_components(graph)
    node_to_comp_size = {}
    for comp in connected_components:
        sz = len(comp)
        for node in comp:
            node_to_comp_size[node] = sz

    # For each subgraph, record the component size (use first node — all nodes share the same component)
    comp_sizes = np.array([node_to_comp_size[indices[0]] for indices in subgraph_indices])

    # Extract coordinates for each subgraph
    subgraph_coords = coordinates[subgraph_indices]

    n_exact = int((comp_sizes == subgraph_size).sum())
    logger.info(f"Found {len(subgraph_indices)} connected subgraphs of size {subgraph_size} "
                f"({n_exact} from exact-size components)")

    return torch.tensor(subgraph_coords, dtype=torch.float32), comp_sizes


def rotation_to_align_with_z(normal: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that maps a unit normal vector to [0, 0, 1].

    Uses Rodrigues' rotation formula. If normal is already [0,0,1], returns identity.
    If normal is [0,0,-1], returns a 180-degree rotation around X.

    Args:
        normal: Unit normal vector (3,)

    Returns:
        Rotation matrix (3, 3)
    """
    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(normal, z)

    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # 180-degree rotation around X axis
        return np.diag([1.0, -1.0, -1.0])

    # Rotation axis = normal x z (cross product)
    v = np.cross(normal, z)
    s = np.linalg.norm(v)
    c = dot

    # Skew-symmetric matrix of v
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def align_subgraphs_to_membrane(
    subgraphs: torch.Tensor,
    membrane_positions: np.ndarray,
    membrane_normals: np.ndarray,
) -> torch.Tensor:
    """
    Transform subgraph coordinates to membrane-aligned frame (normal -> Z).

    For each subgraph, finds the nearest membrane vertex for each AuNP,
    averages the normals, and rotates the subgraph so the membrane normal
    aligns with the Z axis.

    Args:
        subgraphs: Tensor of shape (n_subgraphs, n_points, 3)
        membrane_positions: Array of shape (n_vertices, 3) in tomogram nm
        membrane_normals: Array of shape (n_vertices, 3) in tomogram frame

    Returns:
        Transformed subgraphs tensor of same shape
    """
    if len(subgraphs) == 0:
        return subgraphs

    kdt = cKDTree(membrane_positions)
    subgraphs_np = subgraphs.numpy()
    aligned = np.zeros_like(subgraphs_np)

    for i in range(len(subgraphs_np)):
        # Find nearest membrane vertex for each AuNP in this subgraph
        _, idxs = kdt.query(subgraphs_np[i])  # (n_points,)
        local_normals = membrane_normals[idxs]  # (n_points, 3)

        # Average normal for this subgraph
        avg_normal = local_normals.mean(axis=0)
        avg_normal /= np.linalg.norm(avg_normal)

        # Compute rotation that maps normal to Z
        R = rotation_to_align_with_z(avg_normal)

        # Apply rotation to subgraph coordinates
        aligned[i] = subgraphs_np[i] @ R.T

    logger.info(f"Aligned {len(subgraphs)} subgraphs to membrane frame")
    return torch.tensor(aligned, dtype=torch.float32)


def generate_decoy_subgraphs(config) -> torch.Tensor:
    """
    Generate decoy subgraphs from a STAR file (e.g. liquid model AuNP positions).

    Loads coordinates (in Angstroms), converts to nm, adds Gaussian noise,
    builds a proximity graph, and extracts connected subgraphs.

    Args:
        config: AnalysisConfig with decoy_star_file, decoy_noise_sigma, etc.

    Returns:
        Tensor of shape (n_decoys, subgraph_size, 3)
    """
    import starfile
    from pathlib import Path

    star_path = Path(config.decoy_star_file)
    if not star_path.exists():
        logger.error(f"Decoy STAR file not found: {star_path}")
        return torch.empty(0, config.subgraph_size, 3)

    # Load coordinates (in Angstroms)
    data = starfile.read(star_path)
    cols = config.decoy_coordinate_columns
    coords_angstrom = data[cols].values

    # Convert to nm
    coords_nm = coords_angstrom / 10.0
    logger.info(f"Loaded {len(coords_nm)} decoy AuNP positions from {star_path}")

    # Add Gaussian noise
    rng = np.random.default_rng(config.decoy_seed)
    noise = rng.normal(0, config.decoy_noise_sigma, size=coords_nm.shape)
    coords_noisy = coords_nm + noise
    logger.info(f"Applied {config.decoy_noise_sigma} nm Gaussian noise to decoy positions")

    # Build proximity graph and extract subgraphs
    graph, _ = build_proximity_graph(
        coords_noisy,
        max_distance=config.kdt_max_distance,
        min_distance=config.kdt_min_distance
    )
    subgraphs, _ = extract_connected_subgraphs(graph, config.subgraph_size, coords_noisy)

    # Subsample if requested
    if config.decoy_n_subgraphs is not None and len(subgraphs) > config.decoy_n_subgraphs:
        indices = rng.choice(len(subgraphs), config.decoy_n_subgraphs, replace=False)
        indices.sort()
        subgraphs = subgraphs[indices]
        logger.info(f"Subsampled to {len(subgraphs)} decoy subgraphs")

    return subgraphs


def process_tomogram_graphs(
    coordinate_list: List[torch.Tensor],
    config,
    return_graphs: bool = False,
    membrane_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[torch.Tensor, List[Tuple[int]], Optional[List[dict]]]:
    """
    Process all tomograms to extract connected subgraphs.

    Args:
        coordinate_list: List of coordinate tensors, one per tomogram
        config: AnalysisConfig instance with graph parameters
        return_graphs: If True, return graph objects and metadata for visualization
        membrane_data: Optional list of (positions, normals) per tomogram for
            membrane-normal-constrained alignment. If provided, subgraph coordinates
            are transformed to membrane-aligned frame (normal -> Z axis).

    Returns:
        Tuple of:
            - Packed subgraph tensor (n_total_subgraphs, subgraph_size, 3)
            - List of tuples indicating how many subgraphs came from each tomogram
            - (Optional) List of dicts with graph metadata for each tomogram
    """
    import einops

    subgraphs_tensors = []
    original_tensors = []  # Original (tomogram-frame) coordinates for overlay
    comp_size_lists = []   # Component size each subgraph belongs to
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
        subgraphs, subgraph_comp_sizes = extract_connected_subgraphs(
            graph,
            config.subgraph_size,
            coords_np
        )

        # Keep original coordinates before membrane alignment
        original_tensors.append(subgraphs.clone())
        comp_size_lists.append(subgraph_comp_sizes)

        # Transform to membrane-aligned frame if membrane data available
        if membrane_data is not None and len(subgraphs) > 0:
            mem_positions, mem_normals = membrane_data[i]
            subgraphs = align_subgraphs_to_membrane(subgraphs, mem_positions, mem_normals)

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
        original_tensors.append(test_shape)
        comp_size_lists.append(np.array([config.subgraph_size]))  # exact size
        logger.info("Added test shape to subgraphs")

    # Add decoy subgraphs from liquid model if configured
    if config.decoy_star_file:
        decoy_subgraphs = generate_decoy_subgraphs(config)
        if len(decoy_subgraphs) > 0:
            subgraphs_tensors.append(decoy_subgraphs)
            original_tensors.append(decoy_subgraphs)
            # Decoys don't have meaningful component sizes; mark as 0
            comp_size_lists.append(np.zeros(len(decoy_subgraphs), dtype=int))
            logger.info(f"Added {len(decoy_subgraphs)} decoy subgraphs from liquid model")

    # Pack all subgraphs into a single tensor
    packed_tensor, pack_info = einops.pack(subgraphs_tensors, pattern="* p d")
    packed_original, _ = einops.pack(original_tensors, pattern="* p d")
    packed_comp_sizes = np.concatenate(comp_size_lists)

    logger.info(f"Packed {len(subgraphs_tensors)} tomogram subgraphs into tensor of shape {packed_tensor.shape}")
    logger.info(f"Subgraphs per tomogram: {pack_info}")

    return packed_tensor, pack_info, graph_metadata, packed_original, packed_comp_sizes
