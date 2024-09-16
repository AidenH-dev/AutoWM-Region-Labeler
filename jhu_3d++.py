## MEANT TO BE UPDATED VERSION OF jhu_3d+.py by using plotly and dash app to offload computations to the GPU for better modeling performance and interactibility
import nibabel as nib
import numpy as np
from skimage import measure
import plotly.graph_objects as go
from dash import Dash, dash_table, html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import random
import ast
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Look-up table (LUT) for white matter regions, including "Unclassified"
lut = {
    0: "Unclassified",
    1: "Middle_cerebellar_peduncle",
    2: "Pontine_crossing_tract_(a_part_of_MCP)",
    3: "Genu_of_corpus_callosum",
    4: "Body_of_corpus_callosum",
    5: "Splenium_of_corpus_callosum",
    6: "Fornix_(column_and_body_of_fornix)",
    7: "Corticospinal_tract_R",
    8: "Corticospinal_tract_L",
    9: "Medial_lemniscus_R",
    10: "Medial_lemniscus_L",
    11: "Inferior_cerebellar_peduncle_R",
    12: "Inferior_cerebellar_peduncle_L",
    13: "Superior_cerebellar_peduncle_R",
    14: "Superior_cerebellar_peduncle_L",
    15: "Cerebral_peduncle_R",
    16: "Cerebral_peduncle_L",
    17: "Anterior_limb_of_internal_capsule_R",
    18: "Anterior_limb_of_internal_capsule_L",
    19: "Posterior_limb_of_internal_capsule_R",
    20: "Posterior_limb_of_internal_capsule_L",
    21: "Retrolenticular_part_of_internal_capsule_R",
    22: "Retrolenticular_part_of_internal_capsule_L",
    23: "Anterior_corona_radiata_R",
    24: "Anterior_corona_radiata_L",
    25: "Superior_corona_radiata_R",
    26: "Superior_corona_radiata_L",
    27: "Posterior_corona_radiata_R",
    28: "Posterior_corona_radiata_L",
    29: "Posterior_thalamic_radiation_(include_optic_radiation)_R",
    30: "Posterior_thalamic_radiation_(include_optic_radiation)_L",
    31: "Sagittal_stratum_(include_inferior_longitidinal_fasciculus_and_inferior_fronto-occipital_fasciculus)_R",
    32: "Sagittal_stratum_(include_inferior_longitidinal_fasciculus_and_inferior_fronto-occipital_fasciculus)_L",
    33: "External_capsule_R",
    34: "External_capsule_L",
    35: "Cingulum_(cingulate_gyrus)_R",
    36: "Cingulum_(cingulate_gyrus)_L",
    37: "Cingulum_(hippocampus)_R",
    38: "Cingulum_(hippocampus)_L",
    39: "Fornix_(cres)_/_Stria_terminalis_(can_not_be_resolved_with_current_resolution)_R",
    40: "Fornix_(cres)_/_Stria_terminalis_(can_not_be_resolved_with_current_resolution)_L",
    41: "Superior_longitudinal_fasciculus_R",
    42: "Superior_longitudinal_fasciculus_L",
    43: "Superior_fronto-occipital_fasciculus_(could_be_a_part_of_anterior_internal_capsule)_R",
    44: "Superior_fronto-occipital_fasciculus_(could_be_a_part_of_anterior_internal_capsule)_L",
    45: "Uncinate_fasciculus_R",
    46: "Uncinate_fasciculus_L",
    47: "Tapetum_R",
    48: "Tapetum_L"
}

# Pre Step 1: Convert MNI coordinates to MRIcron coordinates
def mni_to_mricron(coords, mni_voxel_size=2):
    scaling_factor = 2.0 / mni_voxel_size  # Adjust for voxel sizes
    converted_coords = []
    for coord in coords:
        X, Y, Z = coord
        X_scaled = X * scaling_factor
        Y_scaled = Y * scaling_factor
        Z_scaled = Z * scaling_factor
        x = int(np.floor((X_scaled + 92) / 2))
        y = int(np.floor((Y_scaled + 128) / 2))
        z = int(np.ceil((Z_scaled + 74) / 2))
        converted_coords.append((x, y, z))
    return converted_coords



# Step 1: Load and extract all regions
def load_and_extract_regions(file_path):
    nii = nib.load(file_path)
    nii_data = nii.get_fdata()

    all_regions_data = []
    all_colors = []
    all_region_names = []

    for region_idx, region_name in lut.items():
        if region_idx == 0:  # Skip "Unclassified"
            continue

        region_data = np.zeros_like(nii_data)
        region_data[nii_data == region_idx] = 1

        if np.any(region_data):
            all_regions_data.append(region_data)
            all_region_names.append(region_name)
            color = (random.random(), random.random(), random.random())
            all_colors.append(color)

    return all_regions_data, all_colors, all_region_names, nii_data

# Step 2: Identify the region based on voxel coordinates
def identify_region(voxel_coords, atlas_data):
    label_value = int(atlas_data[tuple(voxel_coords)])
    return lut.get(label_value, "Unclassified"), label_value

# Step 3: Create a 3D mesh for the region using marching cubes
def create_mesh_for_region(region_data):
    verts, faces, normals, values = measure.marching_cubes(region_data, level=0.5)
    x, y, z = verts.T  # Extract vertices
    i, j, k = faces.T  # Extract faces
    return x, y, z, i, j, k

# Step 4: Create 3D plot with regions and vectors
def create_3d_plot_with_vectors(all_regions_data, all_colors, all_region_names, mni_coords, vectors, atlas_data):
    fig = go.Figure()

    # Plot each white matter region
    for idx, region_data in enumerate(all_regions_data):
        x, y, z, i, j, k = create_mesh_for_region(region_data)
        color_rgb = list(map(float, all_colors[idx]))

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=f'rgb({color_rgb[0]*255},{color_rgb[1]*255},{color_rgb[2]*255})',
            opacity=0.1,
            name=all_region_names[idx]
        ))

    # Plot MNI coordinates and vectors
    for i, coord in enumerate(mni_coords):
        region_name, region_idx = identify_region(coord, atlas_data)

        # Log the coordinate for debugging
        logging.debug(f"MNI Coord {i}: {coord}, Region: {region_name}")

        # Plot MNI coordinate
        fig.add_trace(go.Scatter3d(
            x=[coord[0]], y=[coord[1]], z=[coord[2]],
            mode='markers',
            marker=dict(size=5, color='rgba(255, 0, 0, 1)'),  # Make the point more visible
            name=f'MNI Coord {i} ({region_name})'
        ))

        # Plot vectors to the closest regions
        for j, vec in enumerate(vectors[i]):
            vector = np.array(vec)
            if np.all(vector == 0):
                logging.debug(f"Skipping zero vector for {coord}")
                continue  # Skip zero vectors

            # Calculate the endpoint of the vector
            end_point = np.array(coord) + vector

            fig.add_trace(go.Scatter3d(
                x=[coord[0], end_point[0]],
                y=[coord[1], end_point[1]],
                z=[coord[2], end_point[2]],
                mode='lines',
                line=dict(color='green', width=5),
                name=f'Vector {i}-{j}'
            ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=1.05, y=1, traceorder="normal",
            font=dict(size=5), itemwidth=30, bordercolor="Black", borderwidth=1
        )
    )

    return fig

# Step 5: Create the Dash app
def create_dash_app(all_region_names, fig):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Interactive White Matter Tract Visualization"),
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='region-table',
                    columns=[{'name': 'Region', 'id': 'Region'}],
                    data=[{'Region': region} for region in all_region_names],
                    row_selectable='single',
                    style_cell={'textAlign': 'left'},
                    style_table={'height': '90vh', 'overflowY': 'auto'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                dcc.Graph(
                    id='3d-graph',
                    figure=fig,
                    style={'height': '90vh'}
                )
            ], style={'width': '70%', 'display': 'inline-block'})
        ], style={'display': 'flex', 'height': '100vh'})
    ])

    @app.callback(
        Output('3d-graph', 'figure'),
        [Input('region-table', 'selected_rows')]
    )
    def update_3d_plot(selected_rows):
        for trace in fig['data']:
            trace['opacity'] = 0.1

        if selected_rows:
            selected_region_idx = selected_rows[0]
            fig['data'][selected_region_idx]['opacity'] = 1.0

        return fig

    return app

# Main function
def main():
    # Load white matter regions
    file_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-2mm.nii.gz'
    all_regions_data, all_colors, all_region_names, atlas_data = load_and_extract_regions(file_path)

    # Load CSV data for MNI coordinates and vectors
    csv_file_path = '../AutoWM-Region-Labeler/output.csv'
    data = pd.read_csv(csv_file_path)
    mni_coords = data[['x', 'y', 'z']].values
    vectors = [ 
        [eval(data.iloc[i]['region 1 vector']), 
        eval(data.iloc[i]['region 2 vector']), 
        eval(data.iloc[i]['region 3 vector'])] 
        for i in range(len(data))
]
    # Convert MNI coordinates to MRIcron coordinates
    shifted_coordinates = mni_to_mricron(mni_coords)

    # Create 3D Plotly figure with regions and vectors
    fig = create_3d_plot_with_vectors(all_regions_data, all_colors, all_region_names, shifted_coordinates, vectors, atlas_data)

    # Create and run the Dash app
    app = create_dash_app(all_region_names, fig)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()