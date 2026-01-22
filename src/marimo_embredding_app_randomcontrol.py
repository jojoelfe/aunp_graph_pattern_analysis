import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _(embedding, mo, scatter):

    chart = mo.ui.altair_chart(scatter(embedding),)
    chart
    return (chart,)


@app.cell
def _(
    best_rmsm_indices,
    chart,
    np,
    roma,
    subgraph_centered_coordinates,
    subgraph_centered_coordinates_aligned,
    torch,
):
    def average_structure_plot():
        import plotly.express as px
        import pandas as pd
        import plotly.graph_objects as go
        if len(chart.value.index)<1:
            return
        reference = chart.value.index[0]
        all = chart.value.index
        coordinates_reference = subgraph_centered_coordinates[reference]

        aligned = subgraph_centered_coordinates_aligned[reference,all,best_rmsm_indices[reference,all]]
        average = torch.mean(aligned, dim=0)

        R, _ = roma.utils.rigid_points_registration(aligned, average.unsqueeze(0)) # subgraphs subgraphs permutations  3x3 ritation matrix

        aligned_aligned = aligned @ torch.transpose(R,-1,-2)

        df = pd.DataFrame({
            'x': aligned_aligned[:,:,0].flatten(),
            'y': aligned_aligned[:,:,1].flatten(),
            'z': aligned_aligned[:,:,2].flatten(),
            'c': np.repeat(np.arange(len(aligned)), aligned.shape[1])
        })
        # Set 'c' as a categorical variable
        df['c'] = df['c'].astype('category')


        fig = px.scatter_3d(df,x='x',y='y',z='z',color='c',color_discrete_sequence=px.colors.qualitative.Pastel,
                                size_max=1,opacity=0.8)

        fig.add_trace(go.Scatter3d(
            x=average[:,0],
            y=average[:,1],
            z=average[:,2],

            name='Average',
            mode='markers',
            marker=dict(
                size=10,
                color='red',  # Set color to red
                opacity=0.8,
            ),
        ))
        return fig
    average_structure_plot()

    return


@app.cell(hide_code=True)
def _(
    basefolder,
    chart,
    load_tomogram,
    np,
    plt,
    subgraphs_tensor,
    tomo_ps,
    tomos,
):
    def _():
        from numpy.random import normal, rand

        plt.close()
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for tin, tomor in enumerate(tomos):
            az = basefolder / tomor[0] / "fiducial_tracking" / "active_zonograms" / f"active_zonogram_{tomos[0][1]}.mrc"
            res_ddw, rshape = load_tomogram(az)
            axs.flatten()[tin].imshow(res_ddw.min(dim=0)[0], cmap="gray",interpolation='mitchell',origin="lower")
            axs.flatten()[tin].set_axis_off()
            az_data = np.load(az.with_suffix(".npy") ,allow_pickle=True)
            min_ind = sum([int(s[0]) for s in tomo_ps[:tin]])
            max_ind = min_ind + int(tomo_ps[tin][0])
            for ind, s in enumerate(chart.value.index):
                if s < min_ind or s > max_ind:
                    continue
                coordinates = subgraphs_tensor[s]
                selected_aunp_pos_transformed = (coordinates - az_data.tolist()["center"]) @ az_data.tolist()["cs"].T
                selected_aunp_pos_transformed += np.floor(np.array(rshape)[[2,1,0]]/2)

                color = rand(3)
                jitter = normal(0, 0.5, size=2)  # Adjust the standard deviation as needed
                axs.flatten()[tin].plot(selected_aunp_pos_transformed[:,0]+ jitter[0], selected_aunp_pos_transformed[:,1] + jitter[1], color=color)


        plt.tight_layout()
        return fig


    _()
    return


@app.cell
def _(np, pd, tomo_ps, tsne_result):
    color = np.zeros_like(tsne_result[:, 0]) + 5
    print(color.shape)
    low = 0
    for ir, fs in enumerate(tomo_ps):
        fs = int(fs[0])
        color[low:low+fs] = ir
        low += fs 
        print(low)
        print(fs)
    print(color)
    embedding = pd.DataFrame(
        {"x": tsne_result[:, 0], 
         "y": tsne_result[:, 1], 
         "c": color}
    ).reset_index()
    return (embedding,)


@app.cell
def _(alt, tomo_ps):
    def scatter(df):
        return (alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            color=alt.Color("c:N", scale=alt.Scale(domain=list(range(len(tomo_ps))))),
        ).properties(width=500, height=500))
    return (scatter,)


@app.cell
def _(mo, np, random_points):
    import starfile
    from findingampa.utils.analysis import render_active_zonograms, calculate_probit, estimate_anisotropic_gaussian_for_probit, calculate_1q_p_value, render_active_zonograms, select_aunps
    import pymeshlab
    from pathlib import Path
    from scipy.spatial import cKDTree
    from matplotlib import pyplot as plt
    import torch
    import mrcfile
    import json
    from scipy.spatial import cKDTree
    import rustworkx as rx
    import torch
    import roma
    import einops
    from itertools import permutations
    import pandas as pd

    def load_tomogram(az):
        with mrcfile.open(az, permissive=True) as mrc:
            res_ddw = torch.tensor(mrc.data.copy())

        return res_ddw, res_ddw.shape

    with mo.persistent_cache('rmsd_matrix_02'):
        basefolder = Path("/scratch/pompeii/elferich/gouaux_tomo/graph/graph/")

        tomos = [("20231026_HippAu_14","0"),
                 ("20240111_WaffleHipp_227","0"),
                 ("20240111_WaffleHipp_96","0"),
                 ("20240111_WaffleHipp_116","0"),]
        subgraphs_tensors = []
        for tomo in tomos:
            aunps_data = starfile.read(basefolder / tomo[0] / "fiducial_tracking" /"aunps"/"aunp_tm_BP_active_zone_0.star")
            print(basefolder / tomo[0] / "fiducial_tracking" /"aunps"/"aunp_tm_BP_active_zone_0.star")
            coordinates = aunps_data[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].values 

            kdt = cKDTree(coordinates)
            pairs = kdt.query_pairs(11,output_type="ndarray")
            distances = np.linalg.norm(coordinates[pairs[:,0]] - coordinates[pairs[:,1]], axis=1)
            pairs = pairs[distances>5]
            graph = rx.PyGraph()

            graph.add_nodes_from(range(len(coordinates)))

            for i, j in pairs:
                graph.add_edge(i,j,None)

            graph.edge_list()

            subgraphs_4 = rx.connected_subgraphs(graph,4)
            subgraphs_tensors.append(torch.tensor(coordinates[subgraphs_4]))

        injected_shapes_tensor = torch.tensor([[
            [0.0,0.0,0.0],
            [9.0,0.0,0.0],
            [18.0,0.0,0.0],
            [27.0,0.0,0.0],
        ]])
        subgraphs_tensors.append(injected_shapes_tensor)
        subgraphs_tensors.append(torch.tensor(random_points))
        subgraphs_tensor, tomo_ps = einops.pack(subgraphs_tensors,pattern="* p d")
        print(subgraphs_tensor.shape)
        print(tomo_ps)
        # Subtract the centroid coordinates
        subgraph_centered_coordinates = subgraphs_tensor - subgraphs_tensor.mean(dim=1, keepdim=True) # Subgraphs, Points, Coordinates
        # In the dimension 0, create all pemutations of dimension 2
        perms = list(permutations(range(subgraphs_tensor.shape[1])))
        subgraph_centered_coordinates_permutations = subgraph_centered_coordinates[ :, perms, :] # Subgraphs, Permutations, Points, Coordinates


        left_side = einops.rearrange(subgraph_centered_coordinates_permutations[:,:,:,:], "s p n d -> 1 s p n d") # Permutation subgraph points coordinates
        right_side = einops.rearrange(subgraph_centered_coordinates_permutations[:,[0],:,:], "s p n d -> s 1 p n d") # Permutation subgraph points coordinates
        R, _ = roma.utils.rigid_points_registration(left_side, right_side) # subgraphs subgraphs permutations  3x3 ritation matrix
        subgraph_centered_coordinates_aligned = left_side @ torch.transpose(R,-1,-2)
        print(subgraph_centered_coordinates_aligned.shape)
        # Calculate RMSD between the aligned points and right side
        rmsd = torch.sqrt( ( 1/ subgraph_centered_coordinates_aligned.shape[3]) *
                    torch.sum((subgraph_centered_coordinates_aligned - right_side) ** 2, dim=(-1,-2))
                         ) # subgraphs subgraphs permutations

        best_rmsd, best_rmsm_indices = torch.min(rmsd, dim=2)
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, metric="precomputed", init="random", perplexity=20)
    tsne_result = tsne.fit_transform(best_rmsd)
    from sklearn.decomposition import PCA

    # Perform PCA
    pca = PCA(n_components=2)
    #Flatten last two dimensions

    pca_result = pca.fit_transform(subgraph_centered_coordinates.flatten(start_dim=1).numpy())

    return (
        basefolder,
        best_rmsd,
        best_rmsm_indices,
        load_tomogram,
        pd,
        plt,
        roma,
        subgraph_centered_coordinates,
        subgraph_centered_coordinates_aligned,
        subgraphs_tensor,
        tomo_ps,
        tomos,
        torch,
        tsne_result,
    )


@app.cell
def _(best_rmsd, plt):
    # Show best_rmsd matrix
    # Low values should be brighter
    # Scale from 5 to 0
    # Show scalebar
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(best_rmsd, cmap="viridis_r", interpolation='nearest', origin="lower",vmin=0, vmax=5)
    ax.set_xlabel("Subgraph")
    ax.set_ylabel("Subgraph")
    plt.colorbar(ax.imshow(best_rmsd, cmap="viridis_r", interpolation='nearest', origin="lower",vmin=0, vmax=5), ax=ax)
    fig
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("altair")

    import altair as alt
    return (alt,)


@app.cell
def _(np):
    from scipy.spatial.distance import pdist, squareform
    import networkx as nx

    def generate_valid_points(n_points=4, min_dist=5.0, max_dist=11.0, max_attempts=10000):
        for _ in range(max_attempts):
            points = [np.random.uniform(-max_dist, max_dist, size=3)]  # First point placed anywhere

            while len(points) < n_points:
                candidate = np.random.uniform(-max_dist, max_dist, size=3)
                dists = np.linalg.norm(np.array(points) - candidate, axis=1)

                if np.all(dists >= min_dist) and np.any(dists <= max_dist):
                    points.append(candidate)

            # Check connectivity of the graph
            dist_matrix = squareform(pdist(np.array(points)))
            G = nx.Graph()
            for i in range(n_points):
                for j in range(i+1, n_points):
                    if min_dist <= dist_matrix[i, j] <= max_dist:
                        G.add_edge(i, j)

            if nx.is_connected(G):
                return np.array(points)

        raise RuntimeError("Failed to generate a valid configuration within max_attempts.")

    # Example usage
    random_points = []
    for iq in range(1000):
        random_points.append(generate_valid_points())
    random_points = np.array(random_points)
    return (random_points,)


@app.cell
def _(random_points):
    random_points.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
