"""Alignment and RMSD computation for subgraphs.

Uses Z-axis-only rotation (membrane-normal-constrained alignment).
Subgraph coordinates are expected to be in membrane-aligned frame
where Z = membrane normal direction.
"""

import torch
import einops
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


def z_axis_procrustes(P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find optimal rotation around Z-axis that aligns P to Q.

    Uses the analytical 2D Procrustes solution on XY coordinates,
    then applies the rotation as a 3D Z-axis rotation.

    Args:
        P: Source coordinates (..., n_points, 3) in membrane-aligned frame
        Q: Target coordinates (..., n_points, 3) in membrane-aligned frame

    Returns:
        Tuple of (aligned_P, R) where:
            - aligned_P: P rotated to best align with Q (..., n_points, 3)
            - R: Rotation matrices (..., 3, 3)
    """
    # Extract XY coordinates
    px, py = P[..., 0], P[..., 1]
    qx, qy = Q[..., 0], Q[..., 1]

    # Optimal angle: theta = atan2(sum(px*qy - py*qx), sum(px*qx + py*qy))
    sin_term = (px * qy - py * qx).sum(dim=-1)
    cos_term = (px * qx + py * qy).sum(dim=-1)
    theta = torch.atan2(sin_term, cos_term)

    # Build 3D rotation matrix around Z
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.zeros(*theta.shape, 3, 3, device=P.device, dtype=P.dtype)
    R[..., 0, 0] = cos_t
    R[..., 0, 1] = -sin_t
    R[..., 1, 0] = sin_t
    R[..., 1, 1] = cos_t
    R[..., 2, 2] = 1.0

    # Apply rotation: aligned = P @ R^T
    aligned = P @ torch.transpose(R, -1, -2)

    return aligned, R


def compute_rmsd_matrix(
    subgraph_coords: torch.Tensor,
    chunk_size: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise RMSD matrix using Z-axis-constrained rotation.

    Coordinates should already be in membrane-aligned frame (Z = membrane normal).

    Args:
        subgraph_coords: Tensor of shape (n_subgraphs, n_points, 3)
        chunk_size: Number of rows to process at a time

    Returns:
        Tuple of:
            - best_rmsd: Best RMSD for each pair (n_subgraphs, n_subgraphs)
            - best_indices: Best permutation indices (n_subgraphs, n_subgraphs)
    """
    n_subgraphs, n_points, _ = subgraph_coords.shape

    logger.info(f"Computing RMSD matrix for {n_subgraphs} subgraphs with {n_points} points each")
    logger.info(f"Using Z-axis-constrained rotation, chunk size {chunk_size}")

    # Center coordinates
    centered = center_coordinates(subgraph_coords)

    # Generate all permutations
    perms = generate_all_permutations(n_points)

    # Apply permutations to create all possible point orderings
    # Shape: (n_subgraphs, n_permutations, n_points, 3)
    centered_permuted = centered[:, perms, :]
    logger.info(f"Created permuted coordinates: {centered_permuted.shape}")

    # right_side uses only the first permutation (identity)
    # Shape: (n_subgraphs, 1, 1, n_points, 3)
    right_side = einops.rearrange(
        centered_permuted[:, [0], :, :],
        "s p n d -> s 1 p n d"
    )

    # Allocate output tensors
    best_rmsd = torch.zeros(n_subgraphs, n_subgraphs)
    best_indices = torch.zeros(n_subgraphs, n_subgraphs, dtype=torch.long)

    # Process in chunks of rows
    n_chunks = (n_subgraphs + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_subgraphs)

        logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks} (rows {start}-{end})")

        # left_side for this chunk: (1, chunk_len, n_permutations, n_points, 3)
        chunk_left = einops.rearrange(
            centered_permuted[start:end],
            "s p n d -> 1 s p n d"
        )

        # Compute optimal Z-axis rotation and aligned coordinates
        aligned, _ = z_axis_procrustes(chunk_left, right_side)

        # Compute 3D RMSD: (n_subgraphs, chunk_len, n_permutations)
        rmsd = torch.sqrt(
            (1 / n_points) * torch.sum((aligned - right_side) ** 2, dim=(-1, -2))
        )

        # Find best permutation for each pair in this chunk
        chunk_best_rmsd, chunk_best_indices = torch.min(rmsd, dim=2)

        # Store results (transpose to match [row, col] layout)
        best_rmsd[start:end, :] = chunk_best_rmsd.T
        best_indices[start:end, :] = chunk_best_indices.T

        # Free memory
        del aligned, rmsd, chunk_left, chunk_best_rmsd, chunk_best_indices

    logger.info(f"RMSD matrix computed. Shape: {best_rmsd.shape}")
    logger.info(f"RMSD range: [{best_rmsd.min():.3f}, {best_rmsd.max():.3f}]")

    return best_rmsd, best_indices


def align_selection_to_reference(
    centered_coords: torch.Tensor,
    best_indices: torch.Tensor,
    reference_idx: int,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute aligned coordinates for a selection of subgraphs relative to a reference.

    Uses Z-axis-only rotation (membrane-normal-constrained).

    Args:
        centered_coords: Centered coordinates (n_subgraphs, n_points, 3)
        best_indices: Best permutation indices from RMSD computation (n_subgraphs, n_subgraphs)
        reference_idx: Index of the reference subgraph
        selected_indices: Tensor of indices to align to the reference

    Returns:
        Aligned coordinates (n_selected, n_points, 3)
    """
    n_points = centered_coords.shape[1]
    perms = list(permutations(range(n_points)))

    # Get the reference (identity permutation, centered)
    reference = centered_coords[reference_idx].unsqueeze(0)  # (1, n_points, 3)

    # For each selected subgraph, apply its best permutation then align
    # best_indices[i, j] = which permutation of i best aligns to j
    # So best_indices[idx, reference_idx] = permutation of idx that matches reference
    aligned_list = []
    for idx in selected_indices:
        idx = int(idx)
        perm_idx = int(best_indices[idx, reference_idx])
        perm = perms[perm_idx]

        # Apply best permutation
        permuted = centered_coords[idx][list(perm)]  # (n_points, 3)

        # Compute optimal Z-axis rotation to align to reference
        aligned, _ = z_axis_procrustes(
            permuted.unsqueeze(0),  # (1, n_points, 3)
            reference               # (1, n_points, 3)
        )

        aligned_list.append(aligned.squeeze(0))

    return torch.stack(aligned_list)
