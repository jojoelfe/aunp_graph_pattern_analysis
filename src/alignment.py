"""Alignment and RMSD computation for subgraphs."""

import torch
import einops
import roma
from itertools import permutations
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def center_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Center coordinates by subtracting the centroid.

    Args:
        coordinates: Tensor of shape (n_subgraphs, n_points, 3)

    Returns:
        Centered coordinates tensor of same shape
    """
    centered = coordinates - coordinates.mean(dim=1, keepdim=True)
    logger.info(f"Centered coordinates, shape: {centered.shape}")
    return centered


def generate_all_permutations(n_points: int) -> list:
    """
    Generate all permutations of point indices.

    Args:
        n_points: Number of points

    Returns:
        List of all permutations
    """
    perms = list(permutations(range(n_points)))
    logger.info(f"Generated {len(perms)} permutations for {n_points} points")
    return perms


def compute_rmsd_matrix(
    subgraph_coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute pairwise RMSD matrix for all subgraphs.

    This is the most computationally expensive operation. It:
    1. Centers all subgraph coordinates
    2. Generates all possible point permutations
    3. Performs rigid registration for all pairs and permutations
    4. Computes RMSD and finds best alignment

    Args:
        subgraph_coords: Tensor of shape (n_subgraphs, n_points, 3)

    Returns:
        Tuple of:
            - best_rmsd: Best RMSD for each pair (n_subgraphs, n_subgraphs)
            - best_indices: Best permutation indices (n_subgraphs, n_subgraphs)
            - aligned_coords: Aligned coordinates (n_subgraphs, n_subgraphs, n_permutations, n_points, 3)
    """
    n_subgraphs, n_points, _ = subgraph_coords.shape

    logger.info(f"Computing RMSD matrix for {n_subgraphs} subgraphs with {n_points} points each")
    logger.info("This may take a while...")

    # Center coordinates
    centered = center_coordinates(subgraph_coords)

    # Generate all permutations
    perms = generate_all_permutations(n_points)

    # Apply permutations to create all possible point orderings
    # Shape: (n_subgraphs, n_permutations, n_points, 3)
    centered_permuted = centered[:, perms, :]

    logger.info(f"Created permuted coordinates: {centered_permuted.shape}")

    # Prepare for pairwise comparison
    # left_side: (1, n_subgraphs, n_permutations, n_points, 3)
    left_side = einops.rearrange(
        centered_permuted,
        "s p n d -> 1 s p n d"
    )

    # right_side: (n_subgraphs, 1, 1, n_points, 3) - only first permutation
    right_side = einops.rearrange(
        centered_permuted[:, [0], :, :],
        "s p n d -> s 1 p n d"
    )

    logger.info("Performing rigid point registration...")

    # Compute optimal rotation for all pairs and permutations
    # R: rotation matrices (n_subgraphs, n_subgraphs, n_permutations, 3, 3)
    R, _ = roma.utils.rigid_points_registration(left_side, right_side)

    # Apply rotations
    aligned = left_side @ torch.transpose(R, -1, -2)

    logger.info("Computing RMSD values...")

    # Compute RMSD
    # Shape: (n_subgraphs, n_subgraphs, n_permutations)
    rmsd = torch.sqrt(
        (1 / n_points) * torch.sum((aligned - right_side) ** 2, dim=(-1, -2))
    )

    # Find best permutation for each pair
    best_rmsd, best_indices = torch.min(rmsd, dim=2)

    logger.info(f"RMSD matrix computed. Shape: {best_rmsd.shape}")
    logger.info(f"RMSD range: [{best_rmsd.min():.3f}, {best_rmsd.max():.3f}]")

    return best_rmsd, best_indices, aligned


def get_aligned_coordinates_for_selection(
    subgraph_coords: torch.Tensor,
    best_indices: torch.Tensor,
    reference_idx: int,
    selected_indices: list
) -> torch.Tensor:
    """
    Get aligned coordinates for a selection of subgraphs relative to a reference.

    Args:
        subgraph_coords: Original centered coordinates (n_subgraphs, n_points, 3)
        best_indices: Best permutation indices from RMSD computation
        reference_idx: Index of reference subgraph
        selected_indices: List of indices to align

    Returns:
        Aligned coordinates tensor (n_selected, n_points, 3)
    """
    # This function would need the full aligned coordinates or recompute
    # For now, returning a placeholder
    # In practice, you might want to save the full aligned tensor
    # or recompute on demand
    pass
