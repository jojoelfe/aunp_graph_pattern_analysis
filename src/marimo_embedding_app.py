import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    # Disable tqdm ipywidgets to avoid widget-related issues
    os.environ['TQDM_DISABLE'] = '1'

    import marimo as mo
    import numpy as np
    import torch
    import pandas as pd
    from pathlib import Path
    import json
    return mo, np, torch, pd, Path, json, os


@app.cell
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip
        await micropip.install("altair")

    import altair as alt
    return (alt,)


@app.cell
def _(mo, Path, os):
    """Load configuration and precomputed data."""

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config import AnalysisConfig
    from data_loading import load_precomputed_data

    # Load configuration (use AUNP_CONFIG env var to override, e.g. config_h4kcys.toml)
    config_path = Path(os.environ.get("AUNP_CONFIG", "config.toml"))
    config = AnalysisConfig.from_toml_or_default(config_path)

    # Load precomputed data
    mo.md("## Loading Precomputed Data")

    from alignment import align_selection_to_reference, z_axis_procrustes

    try:
        subgraphs = load_precomputed_data(config.get_precomputed_path("subgraphs.pt"))
        subgraphs_original = load_precomputed_data(config.get_precomputed_path("subgraphs_original.pt"))
        centered_coords = load_precomputed_data(config.get_precomputed_path("centered_coords.pt"))
        best_rmsd = load_precomputed_data(config.get_precomputed_path("rmsd_matrix.pt"))
        best_indices = load_precomputed_data(config.get_precomputed_path("best_indices.pt"))

        # Load metadata
        with open(config.get_precomputed_path("metadata.json")) as f:
            metadata = json.load(f)

        mo.md(f"""
        ✓ Successfully loaded precomputed data:
        - {len(subgraphs)} subgraphs
        - {len(metadata['tomo_names'])} tomograms
        - RMSD matrix shape: {best_rmsd.shape}
        """)

        data_loaded = True
    except FileNotFoundError as e:
        mo.md(f"""
        ❌ **Error**: Precomputed data not found!

        Please run the precomputation step first:
        ```bash
        uv run python src/precompute.py --stage all
        ```

        Error: {e}
        """)
        data_loaded = False

    return (
        config, subgraphs, subgraphs_original, centered_coords, best_rmsd, best_indices,
        align_selection_to_reference, z_axis_procrustes, metadata, data_loaded, load_precomputed_data
    )


@app.cell
def _(mo, data_loaded):
    """Check if data is loaded before proceeding."""
    if not data_loaded:
        mo.stop(True, "Data not loaded. Please run precomputation first.")
    return


@app.cell
def _(mo, np, best_rmsd, metadata):
    """Compute embeddings with user controls."""

    mo.md("## Embedding Configuration")

    # User controls for embedding
    embedding_method = mo.ui.dropdown(
        options=["t-SNE", "PCA"],
        value="t-SNE",
        label="Embedding method:"
    )

    tsne_perplexity = mo.ui.slider(
        start=5, stop=50, value=20, step=5,
        label="t-SNE Perplexity:"
    )

    compute_button = mo.ui.button(
        label="Compute Embedding",
        on_click=lambda _: None
    )

    mo.hstack([embedding_method, tsne_perplexity, compute_button], justify="start")

    return embedding_method, tsne_perplexity, compute_button


@app.cell
def _(
    mo, np, pd, torch, best_rmsd, centered_coords, metadata,
    embedding_method, tsne_perplexity, compute_button
):
    """Compute the selected embedding."""

    # Trigger computation when button is clicked
    compute_button.value

    with mo.status.spinner(title="Computing embedding..."):
        if embedding_method.value == "t-SNE":
            from sklearn.manifold import TSNE

            # Convert to numpy if needed
            rmsd_np = best_rmsd.numpy() if isinstance(best_rmsd, torch.Tensor) else best_rmsd

            tsne = TSNE(
                n_components=2,
                metric="precomputed",
                init="random",
                perplexity=tsne_perplexity.value,
                random_state=42,
                verbose=0
            )
            _embedding_coords = tsne.fit_transform(rmsd_np)

        else:  # PCA
            from sklearn.decomposition import PCA

            # Flatten coordinates for PCA
            coords_flat = centered_coords.flatten(start_dim=1).numpy()

            pca = PCA(n_components=2)
            _embedding_coords = pca.fit_transform(coords_flat)

            mo.md(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Create color labels by tomogram
    _tomo_pack_info = metadata['tomo_pack_info']
    _color = np.zeros(len(_embedding_coords))
    low = 0
    for ir, (fs,) in enumerate(_tomo_pack_info):
        fs = int(fs)
        _color[low:low + fs] = ir
        low += fs

    # Create tomogram name mapping (handle test shape and decoys)
    _tomo_names = metadata['tomo_names'].copy()
    _extra_labels = ["Test Shape", "Liquid Decoy"]
    for i in range(len(_tomo_pack_info) - len(_tomo_names)):
        _tomo_names.append(_extra_labels[i] if i < len(_extra_labels) else f"Extra {i}")

    # Create embedding dataframe
    _is_decoy = np.array([_tomo_names[int(c)] == "Liquid Decoy" for c in _color])

    # Mark subgraphs from exact-size components (component_size == subgraph_size)
    _comp_sizes = np.array(metadata.get("component_sizes", []))
    _subgraph_size = metadata.get("subgraph_size", 4)
    _is_exact = _comp_sizes == _subgraph_size if len(_comp_sizes) == len(_embedding_coords) else np.zeros(len(_embedding_coords), dtype=bool)

    # Compute size category for visual encoding
    _size_cat = np.where(_is_decoy, "decoy", np.where(_is_exact, "exact_size", "regular"))

    embedding = pd.DataFrame({
        "x": _embedding_coords[:, 0],
        "y": _embedding_coords[:, 1],
        "c": _color,
        "tomogram": [_tomo_names[int(c)] for c in _color],
        "is_decoy": _is_decoy,
        "is_exact_size": _is_exact,
        "comp_size": _comp_sizes if len(_comp_sizes) == len(_embedding_coords) else np.zeros(len(_embedding_coords), dtype=int),
        "size_category": _size_cat,
    }).reset_index()

    mo.md(f"""
    ✓ {embedding_method.value} embedding computed

    Embedding shape: {embedding.shape} ({int(_is_decoy.sum())} decoys, {int((~_is_decoy).sum())} real, {int(_is_exact.sum())} exact-size-{_subgraph_size})
    """)

    return embedding


@app.cell
def _(alt, embedding):
    """Create interactive scatter plot with decoys as background."""

    def scatter(df):
        brush = alt.selection_interval()

        return (alt.Chart(df)
            .mark_point(filled=True)
            .encode(
                x=alt.X("x:Q", title="Dimension 1"),
                y=alt.Y("y:Q", title="Dimension 2"),
                color=alt.condition(
                    brush,
                    alt.Color("tomogram:N", legend=alt.Legend(title="Tomogram")),
                    alt.value('lightgray')
                ),
                size=alt.Size(
                    "size_category:N",
                    scale=alt.Scale(domain=["decoy", "regular", "exact_size"], range=[15, 80, 160]),
                    legend=None,
                ),
                opacity=alt.condition(
                    alt.datum.is_decoy,
                    alt.value(0.3),
                    alt.value(0.8),
                ),
                shape=alt.condition(
                    alt.datum.is_exact_size,
                    alt.value("diamond"),
                    alt.value("circle"),
                ),
                strokeWidth=alt.condition(
                    alt.datum.is_exact_size,
                    alt.value(2),
                    alt.value(0),
                ),
                stroke=alt.condition(
                    alt.datum.is_exact_size,
                    alt.value("black"),
                    alt.value("transparent"),
                ),
                tooltip=["index:Q", "tomogram:N", "x:Q", "y:Q", "comp_size:Q", "is_exact_size:N"]
            ).properties(
                width=700,
                height=700,
                title="Subgraph Embedding (diamonds = exact size-4 components; small gray = liquid decoys)"
            ).add_params(brush)
        )

    return (scatter,)


@app.cell
def _(mo, scatter, embedding):
    """Display interactive scatter plot."""

    chart = mo.ui.altair_chart(scatter(embedding))

    mo.vstack([
        mo.md("## Interactive Embedding Visualization"),
        chart
    ])

    return (chart,)


@app.cell
def _(mo, chart, embedding):
    """Show selected points."""

    _selected_df = chart.apply_selection(embedding)
    # Filter out decoys from selection
    selected_real = _selected_df[~_selected_df["is_decoy"]]

    if len(selected_real) > 0 and len(selected_real) < len(embedding[~embedding["is_decoy"]]):
        mo.md(f"""
        **Selected {len(selected_real)} subgraph(s)**

        Indices: {list(selected_real["index"].values[:10])}{'...' if len(selected_real) > 10 else ''}
        """)
    else:
        mo.md("*Drag a box to select subgraphs*")

    return (selected_real,)


@app.cell
def _(mo, np, pd, torch, selected_real, embedding, centered_coords, align_selection_to_reference, z_axis_procrustes, best_indices):
    """3D average structure visualization for selected subgraphs."""

    import plotly.express as px
    import plotly.graph_objects as go

    _has_selection = len(selected_real) > 0 and len(selected_real) < len(embedding[~embedding["is_decoy"]])

    if not _has_selection:
        _output_3d = mo.vstack([
            mo.md("## Average Structure (3D)"),
            mo.md("*Select points in the scatter plot to see average structure*")
        ])
    else:
        try:
            selected = selected_real["index"].values
            reference = selected[0]

            # Get aligned coordinates for selected subgraphs (computed on-the-fly)
            aligned = align_selection_to_reference(
                centered_coords, best_indices, reference, selected
            )
            average = torch.mean(aligned, dim=0)

            # Further align to average using Z-axis-only rotation
            aligned_aligned, _ = z_axis_procrustes(aligned, average.unsqueeze(0))

            # Create dataframe for plotting
            df = pd.DataFrame({
                'x': aligned_aligned[:, :, 0].flatten(),
                'y': aligned_aligned[:, :, 1].flatten(),
                'z': aligned_aligned[:, :, 2].flatten(),
                'subgraph': np.repeat(np.arange(len(aligned)), aligned.shape[1])
            })
            df['subgraph'] = df['subgraph'].astype('category')

            # Create 3D scatter plot
            _plot_fig = px.scatter_3d(
                df, x='x', y='y', z='z',
                color='subgraph',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                opacity=0.7,
                title=f"Aligned Structures ({len(selected)} selected)"
            )

            # Add average structure
            _plot_fig.add_trace(go.Scatter3d(
                x=average[:, 0],
                y=average[:, 1],
                z=average[:, 2],
                name='Average',
                mode='markers',
                marker=dict(size=10, color='red', opacity=0.9)
            ))

            _plot_fig.update_layout(
                height=600,
                scene=dict(aspectmode='data'),
            )

            _output_3d = mo.vstack([
                mo.md("## Average Structure (3D)"),
                _plot_fig
            ])
        except Exception as e:
            _output_3d = mo.vstack([
                mo.md("## Average Structure (3D)"),
                mo.md(f"Error creating 3D visualization: {str(e)}")
            ])

    _output_3d

    return


@app.cell
def _(mo, plt, best_rmsd, config):
    """RMSD heatmap visualization."""

    _rmsd_fig, _rmsd_ax = plt.subplots(figsize=(10, 10))
    _rmsd_im = _rmsd_ax.imshow(
        best_rmsd,
        cmap="viridis_r",
        interpolation='nearest',
        origin="lower",
        vmin=0,
        vmax=config.rmsd_max_value
    )
    _rmsd_ax.set_xlabel("Subgraph Index")
    _rmsd_ax.set_ylabel("Subgraph Index")
    _rmsd_ax.set_title("Pairwise RMSD Matrix")
    plt.colorbar(_rmsd_im, ax=_rmsd_ax, label="RMSD (Å)")

    mo.vstack([
        mo.md("## RMSD Matrix Heatmap"),
        _rmsd_fig
    ])

    return


@app.cell
def _(mo, np, torch, plt, selected_real, embedding, subgraphs_original, metadata, config):
    """Tomogram overlay visualization for selected subgraphs in all active zones."""

    import mrcfile
    import warnings
    import math
    from numpy.random import normal, rand

    _has_selection = len(selected_real) > 0 and len(selected_real) < len(embedding[~embedding["is_decoy"]])

    if not _has_selection:
        _output_tomo = mo.vstack([
            mo.md("## Tomogram Overlay"),
            mo.md("*Select points in the scatter plot to see tomogram overlays*")
        ])
    else:
        _selected_indices = selected_real["index"].values
        _tomos = [tuple(t) for t in metadata['tomos']]
        _tomo_pack_info = metadata['tomo_pack_info']
        _n_tomos = len(_tomos)

        # Dynamic grid layout
        _ncols = min(_n_tomos, 4)
        _nrows = math.ceil(_n_tomos / _ncols)
        _fig, _axs = plt.subplots(_nrows, _ncols, figsize=(5 * _ncols, 5 * _nrows))
        _axs_flat = np.array(_axs).flatten() if _n_tomos > 1 else [_axs]

        # Hide unused panels
        for _ax in _axs_flat[_n_tomos:]:
            _ax.set_visible(False)

        for _tin, _tomo in enumerate(_tomos):
            _ax = _axs_flat[_tin]

            # Get active zonogram path
            _az_path = config.get_active_zonogram_path(_tomo)

            if not _az_path.exists():
                _ax.text(0.5, 0.5, "AZ not found", ha='center', va='center', transform=_ax.transAxes)
                _ax.set_axis_off()
                continue

            try:
                # Load tomogram
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    with mrcfile.open(str(_az_path), permissive=True) as _mrc:
                        _az_data = torch.tensor(_mrc.data.copy())

                # Load active zonogram metadata
                _meta_path = _az_path.with_suffix(".npy")
                if not _meta_path.exists():
                    _ax.text(0.5, 0.5, "Metadata not found", ha='center', va='center', transform=_ax.transAxes)
                    _ax.set_axis_off()
                    continue

                _az_meta = np.load(_meta_path, allow_pickle=True).tolist()

                # Display minimum-intensity projection
                _ax.imshow(
                    _az_data.min(dim=0)[0],
                    cmap="gray",
                    interpolation='mitchell',
                    origin="lower"
                )
                _ax.set_axis_off()

                # Get subgraph index range for this tomogram
                _min_ind = sum([int(s[0]) for s in _tomo_pack_info[:_tin]])
                _max_ind = _min_ind + int(_tomo_pack_info[_tin][0])

                # Count how many selected subgraphs fall in this tomogram
                _n_selected_here = 0
                for _s in _selected_indices:
                    if _s < _min_ind or _s >= _max_ind:
                        continue

                    _coords = subgraphs_original[_s]
                    _coords_t = (_coords.numpy() - _az_meta["center"]) @ _az_meta["cs"].T
                    _coords_t += np.floor(np.array(_az_data.shape)[[2, 1, 0]] / 2)

                    _c = rand(3)
                    _n_pts = len(_coords_t)

                    # Draw thin lines between all pairs (complete graph)
                    from itertools import combinations
                    for _p, _q in combinations(range(_n_pts), 2):
                        _ax.plot(
                            [_coords_t[_p, 0], _coords_t[_q, 0]],
                            [_coords_t[_p, 1], _coords_t[_q, 1]],
                            '-', color=_c, linewidth=0.5, alpha=0.4
                        )

                    # Find minimum-distance perfect matching (2 pairs for 4 points)
                    # All 3 possible pairings: (01+23), (02+13), (03+12)
                    if _n_pts == 4:
                        _pairings = [
                            [(0, 1), (2, 3)],
                            [(0, 2), (1, 3)],
                            [(0, 3), (1, 2)],
                        ]
                        _best_cost = float('inf')
                        _best_pairs = _pairings[0]
                        for _pr in _pairings:
                            _cost = sum(
                                np.linalg.norm(_coords_t[a] - _coords_t[b])
                                for a, b in _pr
                            )
                            if _cost < _best_cost:
                                _best_cost = _cost
                                _best_pairs = _pr

                        for _a, _b in _best_pairs:
                            _ax.plot(
                                [_coords_t[_a, 0], _coords_t[_b, 0]],
                                [_coords_t[_a, 1], _coords_t[_b, 1]],
                                '-', color=_c, linewidth=3
                            )

                    _n_selected_here += 1

                _ax.set_title(f"{metadata['tomo_names'][_tin]} ({_n_selected_here} sel)")

            except Exception as _e:
                _ax.text(0.5, 0.5, f"Error:\n{str(_e)}", ha='center', va='center',
                        transform=_ax.transAxes, fontsize=8, wrap=True)
                _ax.set_axis_off()

        plt.tight_layout()

        _output_tomo = mo.vstack([
            mo.md("## Tomogram Overlay"),
            _fig
        ])

    _output_tomo

    return


@app.cell
def _():
    """Import remaining dependencies."""
    from matplotlib import pyplot as plt
    return (plt,)


if __name__ == "__main__":
    app.run()
