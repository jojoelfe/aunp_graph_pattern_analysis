"""Data loading utilities for graph pattern analysis."""

import torch
import numpy as np
import mrcfile
import starfile
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_tomogram(path: Path) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Load a tomogram from an MRC file.

    Args:
        path: Path to the MRC file

    Returns:
        Tuple of (tomogram data as tensor, shape)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be read
    """
    if not path.exists():
        raise FileNotFoundError(f"Tomogram file not found: {path}")

    try:
        with mrcfile.open(path, permissive=True) as mrc:
            data = torch.tensor(mrc.data.copy())
        logger.info(f"Loaded tomogram from {path}, shape: {data.shape}")
        return data, data.shape
    except Exception as e:
        raise ValueError(f"Failed to load tomogram from {path}: {e}")


def load_aunp_coordinates(star_path: Path) -> np.ndarray:
    """
    Load AUNP coordinates from a STAR file.

    Args:
        star_path: Path to the STAR file

    Returns:
        Numpy array of shape (n_points, 3) with XYZ coordinates

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be read or doesn't have expected columns
    """
    if not star_path.exists():
        raise FileNotFoundError(f"STAR file not found: {star_path}")

    try:
        aunps_data = starfile.read(star_path)
        required_cols = ["faCoordinateX", "faCoordinateY", "faCoordinateZ"]

        if not all(col in aunps_data.columns for col in required_cols):
            raise ValueError(f"STAR file missing required columns: {required_cols}")

        coordinates = aunps_data[required_cols].values
        logger.info(f"Loaded {len(coordinates)} AUNP coordinates from {star_path}")
        return coordinates

    except Exception as e:
        raise ValueError(f"Failed to load AUNP coordinates from {star_path}: {e}")


def load_active_zonogram_metadata(npy_path: Path) -> dict:
    """
    Load active zonogram metadata (center and coordinate system).

    Args:
        npy_path: Path to the .npy metadata file

    Returns:
        Dictionary with 'center' and 'cs' keys

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not npy_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {npy_path}")

    try:
        metadata = np.load(npy_path, allow_pickle=True).tolist()
        logger.info(f"Loaded metadata from {npy_path}")
        return metadata
    except Exception as e:
        raise ValueError(f"Failed to load metadata from {npy_path}: {e}")


def load_all_coordinates(config) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Load coordinates from all tomograms in the configuration.

    Args:
        config: AnalysisConfig instance

    Returns:
        Tuple of (list of coordinate tensors, list of tomogram names)
    """
    coordinate_tensors = []
    tomo_names = []

    for tomo in config.tomos:
        star_path = config.get_aunp_star_path(tomo)
        logger.info(f"Loading coordinates from {star_path}")

        coordinates = load_aunp_coordinates(star_path)
        coordinate_tensors.append(torch.tensor(coordinates, dtype=torch.float32))
        tomo_names.append(f"{tomo[0]}_{tomo[1]}")

    return coordinate_tensors, tomo_names


def save_precomputed_data(path: Path, data: torch.Tensor) -> None:
    """
    Save precomputed tensor data.

    Args:
        path: Path to save the tensor
        data: Tensor to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    logger.info(f"Saved precomputed data to {path}, shape: {data.shape}")


def load_precomputed_data(path: Path) -> torch.Tensor:
    """
    Load precomputed tensor data.

    Args:
        path: Path to the saved tensor

    Returns:
        Loaded tensor

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Precomputed data not found: {path}")

    data = torch.load(path)
    logger.info(f"Loaded precomputed data from {path}, shape: {data.shape}")
    return data
