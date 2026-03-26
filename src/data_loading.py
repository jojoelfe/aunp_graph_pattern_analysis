"""Data loading utilities for graph pattern analysis."""

import torch
import numpy as np
import mrcfile
import starfile
import struct
import json
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


def load_membrane_mesh(glb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load membrane mesh positions and normals from a GLB file.

    Parses the GLB binary format directly (no external mesh library needed).
    Transforms from GLB coordinate frame to tomogram nm frame:
        tomo = (glb.X, -glb.Z, glb.Y)

    Args:
        glb_path: Path to the .glb mesh file

    Returns:
        Tuple of (positions, normals) as numpy arrays of shape (n_vertices, 3)
        in tomogram nm coordinates.
    """
    if not glb_path.exists():
        raise FileNotFoundError(f"Membrane GLB file not found: {glb_path}")

    with open(glb_path, 'rb') as f:
        # GLB header: magic(4) + version(4) + length(4)
        magic, version, total_len = struct.unpack('<III', f.read(12))

        # JSON chunk: length(4) + type(4) + data
        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        gltf = json.loads(f.read(chunk_len).decode('utf-8'))

        # Binary chunk: length(4) + type(4) + data
        bin_len, bin_type = struct.unpack('<II', f.read(8))
        bin_data = f.read(bin_len)

    def _extract_accessor(accessor_idx):
        acc = gltf['accessors'][accessor_idx]
        bv = gltf['bufferViews'][acc['bufferView']]
        offset = bv.get('byteOffset', 0)
        length = bv['byteLength']
        return np.frombuffer(bin_data[offset:offset + length], dtype=np.float32).reshape(-1, 3)

    # Find POSITION and NORMAL accessor indices from mesh primitive attributes
    mesh = gltf['meshes'][0]
    attrs = mesh['primitives'][0]['attributes']
    positions_glb = _extract_accessor(attrs['POSITION'])
    normals_glb = _extract_accessor(attrs['NORMAL'])

    # Transform from GLB frame to tomogram nm: (glb.X, -glb.Z, glb.Y) * 10
    # GLB positions are in units of 10 nm; multiply by 10 to get tomogram nm
    positions = np.column_stack([positions_glb[:, 0], -positions_glb[:, 2], positions_glb[:, 1]]) * 10.0
    # Normals are unit vectors — axis swap only, no scaling
    normals = np.column_stack([normals_glb[:, 0], -normals_glb[:, 2], normals_glb[:, 1]])

    logger.info(f"Loaded membrane mesh from {glb_path}: {len(positions)} vertices")
    return positions, normals


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
        az_id = tomo[2] if len(tomo) == 3 else tomo[1]
        tomo_names.append(f"{tomo[0]}_{az_id}")

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
