# AUNP Graph Pattern Analysis

Analysis pipeline for gold nanoparticle (AUNP) graph patterns in cryo-electron tomography data.

## Setup

Install dependencies using uv:

```bash
uv sync
```

## Configuration

Edit `config.toml` to customize the analysis parameters:

- **Paths**: Set `basefolder` and `output_folder`
- **Tomograms**: List the tomograms to analyze in `tomos`
- **Graph parameters**: Adjust distance thresholds and subgraph size
- **Embedding parameters**: Configure t-SNE and PCA settings

## Workflow

The precomputation pipeline has two stages that can be run independently:

### 1. Extract Subgraphs (Fast, with diagnostics)

This step loads AUNP coordinates, builds proximity graphs, and extracts connected subgraphs. It also generates diagnostic plots showing the graphs and selected subgraphs for each tomogram.

```bash
uv run python src/precompute.py --stage subgraphs
```

**Outputs:**
- `output/subgraphs.pt`: Extracted subgraph coordinates
- `output/metadata.json`: Metadata about the analysis
- `diagnostics/*_graph_subgraphs.png`: Per-tomogram plots showing highlighted subgraphs
- `diagnostics/*_graph_component_size.png`: Per-tomogram plots showing all components colored by size
- `diagnostics/summary.png`: Summary statistics (AUNP counts, subgraph counts)
- `diagnostics/component_size_distribution_subgraphs.png`: Distribution by count of subgraphs
- `diagnostics/component_size_distribution_aunps.png`: Distribution by count of AUNPs

### 2. Compute RMSD Matrix (Slow, computationally expensive)

This step computes pairwise RMSD between all subgraphs with optimal alignment. This is the most computationally intensive step.

```bash
uv run python src/precompute.py --stage rmsd
```

**Outputs:**
- `output/rmsd_matrix.pt`: Pairwise RMSD matrix
- `output/best_indices.pt`: Best permutation indices for alignment
- `output/aligned_coords.pt`: Aligned coordinates
- `output/centered_coords.pt`: Centered coordinates

### Run All Stages

To run both stages in sequence:

```bash
uv run python src/precompute.py --stage all
# or simply
uv run python src/precompute.py
```

### Command-line Options

- `--config path/to/config.toml`: Use a custom config file
- `--stage {subgraphs,rmsd,all}`: Which stage to run (default: all)
- `--output-dir path/to/output`: Override output directory
- `--no-diagnostics`: Skip generating diagnostic plots
- `--force`: Force recomputation even if files exist
- `-v, --verbose`: Enable verbose logging

### 3. Interactive Exploration

After precomputing, launch the marimo app for interactive visualization:

```bash
uv run marimo run src/marimo_embedding_app.py
```

The app will load the precomputed data and provide interactive visualizations:

**Features:**
- **Embedding method selection**: Choose between t-SNE or PCA
- **Interactive parameter tuning**: Adjust t-SNE perplexity on the fly
- **Clickable scatter plot**: Select subgraphs to visualize
- **3D average structure**: See aligned structures for selected subgraphs
- **RMSD heatmap**: View the full pairwise distance matrix
- **Tomogram overlay**: View selected subgraphs in their original tomogram context

The app loads data instantly (no computation needed) and only computes embeddings on demand.

## Project Structure

```
.
├── config.toml                      # Configuration file
├── pyproject.toml                   # Python dependencies
├── src/
│   ├── config.py                   # Configuration loader
│   ├── data_loading.py             # Data loading utilities
│   ├── graph_construction.py       # Graph building and subgraph extraction
│   ├── alignment.py                # RMSD computation and alignment
│   ├── embeddings.py               # Dimensionality reduction
│   ├── visualization.py            # Diagnostic plots
│   ├── precompute.py               # Precomputation script
│   └── marimo_embedding_app.py     # Interactive marimo app
├── diagnostics/                     # Diagnostic plots (generated)
└── output/                          # Precomputed data (generated)
    ├── subgraphs.pt
    ├── rmsd_matrix.pt
    ├── best_indices.pt
    ├── aligned_coords.pt
    ├── centered_coords.pt
    └── metadata.json
```

## Modules

- **config**: Configuration management with TOML support
- **data_loading**: Load tomograms and STAR files
- **graph_construction**: Build proximity graphs and extract connected subgraphs
- **alignment**: Center coordinates, perform rigid registration, compute RMSD
- **embeddings**: t-SNE and PCA dimensionality reduction
- **visualization**: Diagnostic plots for graph analysis
- **precompute**: Main precomputation script with staged execution
