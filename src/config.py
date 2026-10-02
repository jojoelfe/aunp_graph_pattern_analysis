"""Configuration for graph pattern analysis."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import tomllib
import csv
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for graph pattern analysis."""

    # Base paths
    basefolder: Path = Path("/scratch/pompeii/elferich/gouaux_tomo/graph/graph/")
    output_folder: Path = Path("./output")

    # Tomograms to analyze (tomogram_folder, alignment_dir, active_zone_id)
    tomos: List[Tuple[str, str, str]] = field(default_factory=list)

    # Optional CSV file to load tomograms from (overrides tomos list if set)
    # CSV columns: tomoname, set, alignment_dir, aunp_active_zones
    tomogram_csv: Optional[Path] = None

    # Prefer manually curated star files when available (_manual_refined > _manual > plain)
    use_manual_star: bool = True

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

    # Decoy subgraphs from liquid model (optional)
    decoy_star_file: Optional[str] = None  # Path to STAR file with AuNP coordinates (in Å)
    decoy_noise_sigma: float = 5.0         # Gaussian noise sigma in nm
    decoy_n_subgraphs: Optional[int] = None  # Max decoys to include (None = all)
    decoy_seed: int = 42                   # Random seed for noise and subsampling
    decoy_coordinate_columns: List[str] = field(default_factory=lambda: [
        "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"
    ])

    # Simulated comparison star files
    sim_crystal_star_file: Optional[str] = None
    sim_liquid_star_file: Optional[str] = None

    # Paired AuNP star file (optional, for paired analysis mode)
    paired_star_file: Optional[str] = None

    # Membrane normal constraint
    membrane_type: str = "postsynaptic"  # "postsynaptic" or "presynaptic"

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
        if isinstance(self.tomogram_csv, str):
            self.tomogram_csv = Path(self.tomogram_csv)

        # Load tomograms from CSV if provided (overrides tomos list)
        if self.tomogram_csv is not None:
            self.tomos = self._load_tomos_from_csv(self.tomogram_csv)
        elif self.tomos:
            self.tomos = [tuple(t) if isinstance(t, list) else t for t in self.tomos]

        # Ensure output folder exists
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if self.generate_diagnostics:
            self.diagnostics_folder.mkdir(parents=True, exist_ok=True)

    def _load_tomos_from_csv(self, csv_path: Path) -> List[Tuple[str, str, str]]:
        """Load tomogram list from CSV file.

        Expected columns: tomoname, set, alignment_dir, aunp_active_zones
        Returns list of (tomoname, alignment_dir, az_id) tuples.
        Entries where the aunps directory does not exist are skipped with a warning.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Tomogram CSV not found: {csv_path}")

        tomos = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tomo = (row['tomoname'], row['alignment_dir'], str(row['aunp_active_zones']))
                aunps_dir = self.basefolder / tomo[0] / tomo[1] / "aunps"
                if not aunps_dir.exists():
                    logger.warning(f"Skipping {tomo[0]}/{tomo[1]}: aunps directory not found at {aunps_dir}")
                    continue
                tomos.append(tomo)

        logger.info(f"Loaded {len(tomos)} tomograms from {csv_path}")
        return tomos

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
        if "tomogram_csv" in config_dict:
            config_dict["tomogram_csv"] = Path(config_dict["tomogram_csv"])

        # Convert tomos list (only used when tomogram_csv is not set)
        if "tomos" in config_dict and "tomogram_csv" not in config_dict:
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

    def _get_tracking_folder(self, tomo: tuple) -> Path:
        """Get the tracking folder (either fiducial_tracking or patch_tracking).

        Supports both 2-element tuples (folder, az_id) with auto-detection
        and 3-element tuples (folder, alignment_type, az_id) with explicit type.
        """
        tomo_base = self.basefolder / tomo[0]

        # If 3-element tuple, use the explicit alignment type
        if len(tomo) == 3:
            return tomo_base / tomo[1]

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

    def _get_az_id(self, tomo: tuple) -> str:
        """Get the active zone ID from a tomo tuple (supports 2 or 3 element tuples)."""
        if len(tomo) == 3:
            return tomo[2]
        return tomo[1]

    def get_aunp_star_path(self, tomo: tuple) -> Path:
        """Get path to AUNP STAR file for a tomogram.

        When use_manual_star=True, prefers manually curated files in order:
          _manual_refined.star > _manual.star > plain .star
        """
        tracking_folder = self._get_tracking_folder(tomo)
        az_id = self._get_az_id(tomo)
        aunps_dir = tracking_folder / "aunps"

        if self.use_manual_star:
            for suffix in (f"_manual_refined", f"_manual", ""):
                candidate = aunps_dir / f"aunp_tm_BP_active_zone_{az_id}{suffix}.star"
                if candidate.exists():
                    return candidate
            # Return manual_refined path even if missing — will raise clear error downstream
            return aunps_dir / f"aunp_tm_BP_active_zone_{az_id}_manual_refined.star"

        return aunps_dir / f"aunp_tm_BP_active_zone_{az_id}.star"

    def get_active_zonogram_path(self, tomo: tuple) -> Path:
        """Get path to active zonogram MRC file."""
        tracking_folder = self._get_tracking_folder(tomo)
        az_id = self._get_az_id(tomo)
        return tracking_folder / "active_zonograms" / f"active_zonogram_{az_id}.mrc"

    def get_precomputed_path(self, filename: str) -> Path:
        """Get path to precomputed data file."""
        return self.output_folder / filename

    def get_membrane_glb_path(self, tomo: tuple) -> Path:
        """Get path to membrane GLB mesh file."""
        tracking_folder = self._get_tracking_folder(tomo)
        return tracking_folder / "aunps" / f"{self.membrane_type}membranes.glb"
