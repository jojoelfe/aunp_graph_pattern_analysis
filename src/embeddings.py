"""Dimensionality reduction and embedding utilities."""

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def compute_tsne_embedding(
    distance_matrix: torch.Tensor,
    n_components: int = 2,
    perplexity: int = 20,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute t-SNE embedding from a precomputed distance matrix.

    Args:
        distance_matrix: Pairwise distance matrix (n_samples, n_samples)
        n_components: Number of embedding dimensions (default: 2)
        perplexity: t-SNE perplexity parameter (default: 20)
        random_state: Random seed for reproducibility

    Returns:
        Embedding coordinates (n_samples, n_components)
    """
    logger.info(
        f"Computing t-SNE embedding (perplexity={perplexity}, "
        f"n_components={n_components})..."
    )

    # Convert to numpy if needed
    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.numpy()

    # Ensure the matrix is symmetric (sometimes numerical errors create asymmetry)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    tsne = TSNE(
        n_components=n_components,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=random_state
    )

    embedding = tsne.fit_transform(distance_matrix)

    logger.info(f"t-SNE embedding computed, shape: {embedding.shape}")

    return embedding


def compute_pca_embedding(
    coordinates: torch.Tensor,
    n_components: int = 2
) -> np.ndarray:
    """
    Compute PCA embedding from flattened coordinates.

    Args:
        coordinates: Subgraph coordinates (n_subgraphs, n_points, 3)
        n_components: Number of PCA components (default: 2)

    Returns:
        PCA embedding (n_subgraphs, n_components)
    """
    logger.info(f"Computing PCA embedding (n_components={n_components})...")

    # Flatten last two dimensions
    flattened = coordinates.flatten(start_dim=1).numpy()

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(flattened)

    logger.info(
        f"PCA embedding computed, shape: {embedding.shape}, "
        f"explained variance: {pca.explained_variance_ratio_}"
    )

    return embedding


def create_embedding_dataframe(
    embedding: np.ndarray,
    tomo_pack_info: list,
    include_tomo_labels: bool = True
):
    """
    Create a pandas DataFrame with embedding coordinates and metadata.

    Args:
        embedding: Embedding coordinates (n_samples, 2 or 3)
        tomo_pack_info: List of tuples indicating subgraphs per tomogram
        include_tomo_labels: Whether to add tomogram labels as color column

    Returns:
        pandas DataFrame with columns: x, y, [z], c, index
    """
    import pandas as pd

    n_samples = len(embedding)
    df_dict = {"x": embedding[:, 0], "y": embedding[:, 1]}

    if embedding.shape[1] >= 3:
        df_dict["z"] = embedding[:, 2]

    # Add tomogram color labels
    if include_tomo_labels:
        color = np.zeros(n_samples)
        low = 0
        for ir, (fs,) in enumerate(tomo_pack_info):
            fs = int(fs)
            color[low:low + fs] = ir
            low += fs
        df_dict["c"] = color

    df = pd.DataFrame(df_dict).reset_index()

    logger.info(f"Created embedding DataFrame with {len(df)} samples")

    return df
