import nibabel as nib
import numpy as np
from skimage import measure
import plotly.graph_objects as go
from dash import Dash, dash_table, html, dcc
from dash.dependencies import Input, Output
import random

# Look-up table (LUT) for white matter regions, including "Unclassified" (which will be skipped)
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

# Step 1: Load the NIfTI file and extract regions, skipping "Unclassified"
def load_and_extract_regions(file_path):
    nii = nib.load(file_path)
    nii_data = nii.get_fdata()

    all_regions_data = []
    all_colors = []
    all_region_names = []

    for region_idx, region_name in lut.items():
        if region_idx == 0:  # Skip "Unclassified"
            continue
        
        # Extract region
        region_data = np.zeros_like(nii_data)
        region_data[nii_data == region_idx] = 1

        # If the region is present, add it to the list
        if np.any(region_data):
            all_regions_data.append(region_data)
            all_region_names.append(region_name)

            # Assign random color to the region
            color = (random.random(), random.random(), random.random())
            all_colors.append(color)

    return all_regions_data, all_colors, all_region_names

# Step 2: Use marching cubes to create a 3D mesh for each region
def create_mesh_for_region(region_data):
    verts, faces, normals, values = measure.marching_cubes(region_data, level=0.5)
    x, y, z = verts.T
    i, j, k = faces.T
    return x, y, z, i, j, k

# Step 3: Create Plotly 3D figure
def create_3d_plot(all_regions_data, all_colors, all_region_names):
    fig = go.Figure()

    for idx, region_data in enumerate(all_regions_data):
        x, y, z, i, j, k = create_mesh_for_region(region_data)

        # Ensure the color is properly converted to a float list
        color_rgb = list(map(float, all_colors[idx]))

        # Add 3D surface for the region
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=f'rgb({color_rgb[0]*255},{color_rgb[1]*255},{color_rgb[2]*255})',
            opacity=0.1,
            name=all_region_names[idx],  # Name displayed in the legend
            hovertemplate=(
                'X: %{x}<br>'                     # Display X coordinate
                'Y: %{y}<br>'                     # Display Y coordinate
                'Z: %{z}<br>'                     # Display Z coordinate
            ),
        ))

    # Adjust the legend settings to give more space for long names
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=1.05,  # Position legend outside the plot
            y=1,
            traceorder="normal",
            font=dict(size=10),  # Adjust the font size if needed
            itemwidth=100,  # Increase item width to give more space for names
            bordercolor="Black",
            borderwidth=1,
        )
    )

    return fig

    
# Step 4: Create the Dash app
def create_dash_app(all_region_names, fig):
    app = Dash(__name__)

    # Define the layout of the app
    app.layout = html.Div([
        html.H1("Interactive White Matter Tract Visualization"),
        html.Div([
            # Left side: table
            html.Div([
                dash_table.DataTable(
                    id='region-table',
                    columns=[{'name': 'Region', 'id': 'Region'}],
                    data=[{'Region': region} for region in all_region_names],
                    row_selectable='single',
                    style_cell={'textAlign': 'left'},
                    style_table={'height': '90vh', 'overflowY': 'auto'},  # Adjust table to fit the screen height
                )
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
            
            # Right side: 3D model viewer
            html.Div([
                dcc.Graph(
                    id='3d-graph',
                    figure=fig,
                    style={'height': '90vh'}  # Adjust model viewer to fit the screen height
                )
            ], style={'width': '70%', 'display': 'inline-block'})
        ], style={'display': 'flex', 'height': '100vh'})  # Flexbox layout to align side by side
    ])

    # Define the callback for updating the 3D plot
    @app.callback(
        Output('3d-graph', 'figure'),
        [Input('region-table', 'selected_rows')]
    )
    def update_3d_plot(selected_rows):
        # Reset all opacities to default
        for trace in fig['data']:
            trace['opacity'] = 0.1

        if selected_rows:
            selected_region_idx = selected_rows[0]
            fig['data'][selected_region_idx]['opacity'] = 1.0  # Highlight the selected region

        return fig

    return app

# Main function to load regions, create plot, and run the Dash app
def main():
    file_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz'  # Update with your local path
    all_regions_data, all_colors, all_region_names = load_and_extract_regions(file_path)

    # Create the 3D plot with Plotly
    fig = create_3d_plot(all_regions_data, all_colors, all_region_names)

    # Create and run the Dash app
    app = create_dash_app(all_region_names, fig)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
