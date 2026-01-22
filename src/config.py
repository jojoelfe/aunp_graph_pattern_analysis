"""Configuration for graph pattern analysis."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import tomllib
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for graph pattern analysis."""

    # Base paths
    basefolder: Path = Path("/scratch/pompeii/elferich/gouaux_tomo/graph/graph/")
    output_folder: Path = Path("./output")

    # Tomograms to analyze (tomogram_folder, active_zone_id)
    tomos: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("20231026_HippAu_14", "0"),
        ("20240111_WaffleHipp_227", "0"),
        ("20240111_WaffleHipp_96", "0"),
        ("20240111_WaffleHipp_116", "0"),
    ])

    # Graph construction parameters
    kdt_max_distance: float = 11.0  # Maximum distance for KD-tree query
    kdt_min_distance: float = 5.0   # Minimum distance filter
    subgraph_size: int = 4          # Size of connected subgraphs to find

    # Alignment parameters
    rmsd_max_value: float = 5.0     # Maximum RMSD for visualization

    # Embedding parameters
    tsne_perplexity: int = 20
    tsne_n_components: int = 2
    pca_n_components: int = 2

    # Injected test shape (optional)
    include_test_shape: bool = True
    test_shape_coordinates: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
        [18.0, 0.0, 0.0],
        [27.0, 0.0, 0.0],
    ])

    # Diagnostic options
    generate_diagnostics: bool = True
    diagnostics_folder: Path = Path("./diagnostics")
    n_subgraphs_to_highlight: int = 5

    def __post_init__(self):
        """Initialize default values and ensure paths exist."""
        # Convert string paths to Path objects
        if isinstance(self.basefolder, str):
            self.basefolder = Path(self.basefolder)
        if isinstance(self.output_folder, str):
            self.output_folder = Path(self.output_folder)
        if isinstance(self.diagnostics_folder, str):
            self.diagnostics_folder = Path(self.diagnostics_folder)

        # Convert tomos list to tuples if needed
        if self.tomos:
            self.tomos = [tuple(t) if isinstance(t, list) else t for t in self.tomos]

        # Ensure output folder exists
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if self.generate_diagnostics:
            self.diagnostics_folder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_toml(cls, toml_path: Path) -> "AnalysisConfig":
        """
        Load configuration from a TOML file.

        Args:
            toml_path: Path to the TOML configuration file

        Returns:
            AnalysisConfig instance with values from the file

        Raises:
            FileNotFoundError: If the TOML file doesn't exist
        """
        if not toml_path.exists():
            raise FileNotFoundError(f"Config file not found: {toml_path}")

        logger.info(f"Loading configuration from {toml_path}")

        with open(toml_path, "rb") as f:
            config_dict = tomllib.load(f)

        # Extract the analysis section if it exists
        if "analysis" in config_dict:
            config_dict = config_dict["analysis"]

        # Convert paths
        if "basefolder" in config_dict:
            config_dict["basefolder"] = Path(config_dict["basefolder"])
        if "output_folder" in config_dict:
            config_dict["output_folder"] = Path(config_dict["output_folder"])
        if "diagnostics_folder" in config_dict:
            config_dict["diagnostics_folder"] = Path(config_dict["diagnostics_folder"])

        # Convert tomos list
        if "tomos" in config_dict:
            config_dict["tomos"] = [tuple(t) for t in config_dict["tomos"]]

        return cls(**config_dict)

    @classmethod
    def from_toml_or_default(cls, toml_path: Optional[Path] = None) -> "AnalysisConfig":
        """
        Load configuration from TOML file if it exists, otherwise use defaults.

        Args:
            toml_path: Path to the TOML file (default: config.toml in current directory)

        Returns:
            AnalysisConfig instance
        """
        if toml_path is None:
            toml_path = Path("config.toml")

        if toml_path.exists():
            try:
                return cls.from_toml(toml_path)
            except Exception as e:
                logger.warning(f"Failed to load config from {toml_path}: {e}")
                logger.warning("Using default configuration")
                return cls()
        else:
            logger.info(f"Config file {toml_path} not found, using defaults")
            return cls()

    def _get_tracking_folder(self, tomo: Tuple[str, str]) -> Path:
        """Get the tracking folder (either fiducial_tracking or patch_tracking)."""
        tomo_base = self.basefolder / tomo[0]

        # Try fiducial_tracking first
        fiducial_path = tomo_base / "fiducial_tracking"
        if fiducial_path.exists():
            return fiducial_path

        # Fall back to patch_tracking
        patch_path = tomo_base / "patch_tracking"
        if patch_path.exists():
            return patch_path

        # If neither exists, default to fiducial_tracking (will fail later with clear error)
        logger.warning(f"Neither fiducial_tracking nor patch_tracking found for {tomo[0]}, using fiducial_tracking")
        return fiducial_path

    def get_aunp_star_path(self, tomo: Tuple[str, str]) -> Path:
        """Get path to AUNP STAR file for a tomogram."""
        tracking_folder = self._get_tracking_folder(tomo)
        return tracking_folder / "aunps" / f"aunp_tm_BP_active_zone_{tomo[1]}.star"

    def get_active_zonogram_path(self, tomo: Tuple[str, str]) -> Path:
        """Get path to active zonogram MRC file."""
        tracking_folder = self._get_tracking_folder(tomo)
        return tracking_folder / "active_zonograms" / f"active_zonogram_{tomo[1]}.mrc"

    def get_precomputed_path(self, filename: str) -> Path:
        """Get path to precomputed data file."""
        return self.output_folder / filename
