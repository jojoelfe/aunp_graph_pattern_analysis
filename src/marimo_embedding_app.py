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

        # Load pairwise distance features
        features = np.load(config.get_precomputed_path("features.npy"))
        with open(config.get_precomputed_path("feature_names.json")) as f:
            feature_names = json.load(f)

        # Load RMSD data if available (optional, for alignment viz)
        _rmsd_path = config.get_precomputed_path("rmsd_matrix.pt")
        _indices_path = config.get_precomputed_path("best_indices.pt")
        if _rmsd_path.exists() and _indices_path.exists():
            best_rmsd = load_precomputed_data(_rmsd_path)
            best_indices = load_precomputed_data(_indices_path)
            _has_rmsd = True
        else:
            best_rmsd = None
            best_indices = None
            _has_rmsd = False

        # Load metadata
        with open(config.get_precomputed_path("metadata.json")) as f:
            metadata = json.load(f)

        mo.md(f"""
        Successfully loaded precomputed data:
        - {len(subgraphs)} subgraphs
        - {len(metadata['tomo_names'])} tomograms
        - Features: {features.shape[1]} dimensions ({', '.join(feature_names)})
        - RMSD alignment: {'available' if _has_rmsd else 'not computed (run --stage rmsd)'}
        """)

        data_loaded = True
    except FileNotFoundError as e:
        mo.md(f"""
        **Error**: Precomputed data not found!

        Please run the precomputation step first:
        ```bash
        uv run python src/precompute.py --stage all
        ```

        Error: {e}
        """)
        data_loaded = False

    return (
        config, subgraphs, subgraphs_original, centered_coords,
        features, feature_names, best_rmsd, best_indices,
        align_selection_to_reference, z_axis_procrustes, metadata, data_loaded, load_precomputed_data
    )


@app.cell
def _(mo, data_loaded):
    """Check if data is loaded before proceeding."""
    if not data_loaded:
        mo.stop(True, "Data not loaded. Please run precomputation first.")
    return


@app.cell
def _(mo, np, pd, features, feature_names, metadata):
    """Build feature DataFrame for 3D visualization."""

    # Create color labels by tomogram
    _tomo_pack_info = metadata['tomo_pack_info']
    _color = np.zeros(len(features))
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

    _is_decoy = np.array([_tomo_names[int(c)] == "Liquid Decoy" for c in _color])

    # Mark subgraphs from exact-size components
    _comp_sizes = np.array(metadata.get("component_sizes", []))
    _subgraph_size = metadata.get("subgraph_size", 4)
    _is_exact = _comp_sizes == _subgraph_size if len(_comp_sizes) == len(features) else np.zeros(len(features), dtype=bool)

    _size_cat = np.where(_is_decoy, "decoy", np.where(_is_exact, "exact_size", "regular"))

    # Convert features to display units (angles to degrees)
    _feat_dict = {}
    for _fi, _fn in enumerate(feature_names):
        _vals = features[:, _fi]
        if "theta" in _fn:
            _vals = np.degrees(_vals)
        _feat_dict[_fn] = _vals

    embedding = pd.DataFrame({
        "c": _color,
        "tomogram": [_tomo_names[int(c)] for c in _color],
        "is_decoy": _is_decoy,
        "is_exact_size": _is_exact,
        "comp_size": _comp_sizes if len(_comp_sizes) == len(features) else np.zeros(len(features), dtype=int),
        "size_category": _size_cat,
        **_feat_dict,
    }).reset_index()

    mo.md(f"""
    **Feature space**: {features.shape[1]} dimensions ({', '.join(feature_names)})

    {len(embedding)} subgraphs ({int(_is_decoy.sum())} decoys, {int((~_is_decoy).sum())} real, {int(_is_exact.sum())} exact-size-{_subgraph_size})
    """)

    return (embedding,)


@app.cell
def _(mo, np, embedding):
    """3D scatter plot of pair-geometry features."""

    import plotly.graph_objects as _go

    fig = _go.Figure()

    _decoy_mask = embedding["is_decoy"].values
    _real_mask = ~_decoy_mask

    # Decoy trace (small gray background points)
    _dec = embedding[_decoy_mask]
    if len(_dec) > 0:
        fig.add_trace(_go.Scatter3d(
            x=_dec["theta_rot"],
            y=_dec["theta_cc"],
            z=_dec["d_cc"],
            mode='markers',
            marker=dict(size=2, color='lightgray', opacity=0.2),
            name='Liquid Decoy',
            customdata=_dec["index"].values,
            hovertemplate=(
                "idx=%{customdata}<br>"
                "\u03b8<sub>rot</sub>=%{x:.1f}\u00b0<br>"
                "\u03b8<sub>cc</sub>=%{y:.1f}\u00b0<br>"
                "d<sub>cc</sub>=%{z:.1f} nm<extra></extra>"
            ),
        ))

    # Real subgraphs, one trace per tomogram
    for _tomo_name in sorted(embedding[_real_mask]["tomogram"].unique()):
        _mask = _real_mask & (embedding["tomogram"] == _tomo_name)
        _sub = embedding[_mask]
        _is_ex = _sub["is_exact_size"].values

        fig.add_trace(_go.Scatter3d(
            x=_sub["theta_rot"],
            y=_sub["theta_cc"],
            z=_sub["d_cc"],
            mode='markers',
            marker=dict(
                size=[5 if e else 3 for e in _is_ex],
                symbol=["diamond" if e else "circle" for e in _is_ex],
                opacity=0.8,
            ),
            name=_tomo_name,
            customdata=_sub["index"].values,
            hovertemplate=(
                f"{_tomo_name}<br>"
                "idx=%{customdata}<br>"
                "\u03b8<sub>rot</sub>=%{x:.1f}\u00b0<br>"
                "\u03b8<sub>cc</sub>=%{y:.1f}\u00b0<br>"
                "d<sub>cc</sub>=%{z:.1f} nm<extra></extra>"
            ),
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="\u03b8_rot (\u00b0)",
            yaxis_title="\u03b8_cc (\u00b0)",
            zaxis_title="d_cc (nm)",
        ),
        height=700,
        title="3D Feature Space (diamonds = exact size-4; gray = liquid decoys)",
        legend=dict(itemsizing='constant'),
    )

    mo.vstack([
        mo.md("## 3D Feature Space"),
        fig
    ])

    return


@app.cell
def _(mo, np, embedding):
    """2D projection scatter plots for each pair of features."""

    from plotly.subplots import make_subplots as _make_subplots
    import plotly.graph_objects as _go

    _axes = [
        ("d_1", "d\u2081 (nm)"),
        ("d_2", "d\u2082 (nm)"),
        ("d_cc", "d_cc (nm)"),
        ("theta_rot", "\u03b8_rot (\u00b0)"),
        ("theta_cc", "\u03b8_cc (\u00b0)"),
    ]

    # All pairwise projections: 5 choose 2 = 10
    _pairs = [(i, j) for i in range(len(_axes)) for j in range(i + 1, len(_axes))]
    _ncols = 5
    _nrows = 2

    _proj_fig = _make_subplots(
        rows=_nrows, cols=_ncols,
        subplot_titles=[f"{_axes[a][1].split('(')[0].strip()} vs {_axes[b][1].split('(')[0].strip()}" for a, b in _pairs],
        horizontal_spacing=0.04,
        vertical_spacing=0.1,
    )

    _decoy_mask = embedding["is_decoy"].values
    _real_mask = ~_decoy_mask
    _tomo_names_sorted = sorted(embedding[_real_mask]["tomogram"].unique())

    # Assign consistent colors across subplots
    import plotly.express as _px
    _colors = _px.colors.qualitative.Plotly
    _tomo_color = {name: _colors[i % len(_colors)] for i, name in enumerate(_tomo_names_sorted)}

    for _idx, (_ai, _bi) in enumerate(_pairs):
        _row = _idx // _ncols + 1
        _col = _idx % _ncols + 1
        _is_first = (_idx == 0)
        _xcol, _xlabel = _axes[_ai]
        _ycol, _ylabel = _axes[_bi]

        # Decoys
        _dec = embedding[_decoy_mask]
        if len(_dec) > 0:
            _proj_fig.add_trace(
                _go.Scatter(
                    x=_dec[_xcol], y=_dec[_ycol],
                    mode='markers',
                    marker=dict(size=2, color='lightgray', opacity=0.2),
                    name='Liquid Decoy',
                    showlegend=_is_first,
                    legendgroup='Liquid Decoy',
                ),
                row=_row, col=_col,
            )

        # Real subgraphs per tomogram
        for _tomo_name in _tomo_names_sorted:
            _mask = _real_mask & (embedding["tomogram"] == _tomo_name)
            _sub = embedding[_mask]
            _is_ex = _sub["is_exact_size"].values

            _proj_fig.add_trace(
                _go.Scatter(
                    x=_sub[_xcol], y=_sub[_ycol],
                    mode='markers',
                    marker=dict(
                        size=[7 if e else 4 for e in _is_ex],
                        color=_tomo_color[_tomo_name],
                        opacity=0.7,
                        symbol=["diamond" if e else "circle" for e in _is_ex],
                    ),
                    name=_tomo_name,
                    showlegend=_is_first,
                    legendgroup=_tomo_name,
                ),
                row=_row, col=_col,
            )

        _proj_fig.update_xaxes(title_text=_xlabel, row=_row, col=_col)
        _proj_fig.update_yaxes(title_text=_ylabel, row=_row, col=_col)

    _proj_fig.update_layout(
        height=700,
        title="2D Projections",
        legend=dict(itemsizing='constant'),
    )

    mo.vstack([
        mo.md("## 2D Projections"),
        _proj_fig,
    ])

    return


@app.cell
def _(mo, feature_names, np):
    """Axis selection controls for 2D selection scatter."""

    _labels = {
        "d_1": "d\u2081 (nm)",
        "d_2": "d\u2082 (nm)",
        "d_cc": "d_cc (nm)",
        "theta_rot": "\u03b8_rot (\u00b0)",
        "theta_cc": "\u03b8_cc (\u00b0)",
    }
    _options = {_labels.get(fn, fn): fn for fn in feature_names}
    _label_list = list(_options.keys())

    x_axis_select = mo.ui.dropdown(
        options=_options,
        value=_label_list[0],
        label="X axis:",
    )
    y_axis_select = mo.ui.dropdown(
        options=_options,
        value=_label_list[1],
        label="Y axis:",
    )

    mo.hstack([x_axis_select, y_axis_select], justify="start")

    return x_axis_select, y_axis_select


@app.cell
def _(alt, mo, embedding, x_axis_select, y_axis_select):
    """2D selection scatter plot using Altair brush."""

    _xcol = x_axis_select.value
    _ycol = y_axis_select.value

    _labels = {
        "angle_between_axes": "Angle Between Axes (deg)",
        "angle_of_separation": "Angle of Separation (deg)",
        "d_inter": "Inter-pair Distance (nm)",
    }

    brush = alt.selection_interval()

    _scatter = (alt.Chart(embedding)
        .mark_point(filled=True)
        .encode(
            x=alt.X(f"{_xcol}:Q", title=_labels.get(_xcol, _xcol)),
            y=alt.Y(f"{_ycol}:Q", title=_labels.get(_ycol, _ycol)),
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
            tooltip=["index:Q", "tomogram:N", "d_1:Q", "d_2:Q", "d_cc:Q", "theta_rot:Q", "theta_cc:Q", "comp_size:Q"]
        ).properties(
            width=600,
            height=500,
            title="Drag to select subgraphs (diamonds = exact size-4; small = liquid decoys)"
        ).add_params(brush)
    )

    chart = mo.ui.altair_chart(_scatter)

    mo.vstack([
        mo.md("## Selection"),
        chart,
    ])

    return (chart,)


@app.cell
def _(mo, chart, embedding):
    """Extract selected points from Altair chart."""

    _selected_df = chart.apply_selection(embedding)
    selected_real = _selected_df[~_selected_df["is_decoy"]]

    if len(selected_real) > 0 and len(selected_real) < len(embedding[~embedding["is_decoy"]]):
        mo.md(f"""
        **Selected {len(selected_real)} subgraph(s)**

        Indices: {list(selected_real["index"].values[:10])}{'...' if len(selected_real) > 10 else ''}
        """)
    else:
        mo.md("*Drag a box on the scatter plot to select subgraphs*")

    return (selected_real,)


@app.cell
def _(mo, np, pd, torch, selected_real, embedding, centered_coords, align_selection_to_reference, z_axis_procrustes, best_indices):
    """3D average structure visualization for selected subgraphs."""

    import plotly.express as _px
    import plotly.graph_objects as _go

    _has_selection = len(selected_real) > 0 and len(selected_real) < len(embedding[~embedding["is_decoy"]])
    _has_rmsd = best_indices is not None

    if not _has_selection:
        _output_3d = mo.vstack([
            mo.md("## Selected Structures (3D)"),
            mo.md("*Select points in the scatter plot to see structures*")
        ])
    else:
        try:
            selected = selected_real["index"].values
            reference = selected[0]

            if _has_rmsd:
                # Use RMSD-based permutation alignment
                aligned = align_selection_to_reference(
                    centered_coords, best_indices, reference, selected
                )
            else:
                # No permutation data — align directly with Z-axis rotation
                aligned = centered_coords[selected]

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

            _plot_fig = _px.scatter_3d(
                df, x='x', y='y', z='z',
                color='subgraph',
                color_discrete_sequence=_px.colors.qualitative.Pastel,
                opacity=0.7,
                title=f"{'Aligned' if _has_rmsd else 'Centered'} Structures ({len(selected)} selected)"
            )

            _plot_fig.add_trace(_go.Scatter3d(
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
                mo.md("## Selected Structures (3D)"),
                _plot_fig
            ])
        except Exception as e:
            _output_3d = mo.vstack([
                mo.md("## Selected Structures (3D)"),
                mo.md(f"Error creating 3D visualization: {str(e)}")
            ])

    _output_3d

    return



@app.cell
def _(mo, np, torch, plt, selected_real, embedding, subgraphs, subgraphs_original, metadata, config):
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

                    # Transform original coords to AZ pixel space for drawing
                    _coords = subgraphs_original[_s]
                    _coords_t = (_coords.numpy() - _az_meta["center"]) @ _az_meta["cs"].T
                    _coords_t += np.floor(np.array(_az_data.shape)[[2, 1, 0]] / 2)

                    # Use membrane-aligned coords for matching (same as feature computation)
                    _coords_aligned = subgraphs[_s].numpy()
                    _xy = _coords_aligned[:, :2]

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

                    # Find minimum-distance perfect matching using membrane-aligned XY
                    # (consistent with compute_pairwise_features)
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
                                np.linalg.norm(_xy[a] - _xy[b])
                                for a, b in _pr
                            )
                            if _cost < _best_cost:
                                _best_cost = _cost
                                _best_pairs = _pr

                        # Draw matched pairs on the AZ overlay
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
