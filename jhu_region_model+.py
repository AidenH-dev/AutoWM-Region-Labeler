import nibabel as nib
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import pandas as pd

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

# Step 1: Load and extract only the regions that contain MNI coordinates, with optional downsampling
def load_relevant_regions(file_path, relevant_region_ids, downsample_factor=2):
    nii = nib.load(file_path)
    nii_data = nii.get_fdata()

    selected_regions_data = []
    selected_colors = []

    for region_idx in relevant_region_ids:
        if region_idx == 0:  # Skip "Unclassified" region for 3D model generation
            continue
        # Extract relevant regions only
        region_data = np.zeros_like(nii_data)
        region_data[nii_data == region_idx] = 1

        if np.any(region_data):
            # Downsample to reduce the complexity of the mesh
            region_data = region_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            selected_regions_data.append(region_data)
            # Assign random color to the region
            color = (random.random(), random.random(), random.random())
            selected_colors.append(color)

    return selected_regions_data, selected_colors, nii_data

# Step 2: Identify the white matter region corresponding to MNI coordinates
def identify_region(voxel_coords, atlas_data, lut):
    label_value = int(atlas_data[tuple(voxel_coords)])
    return lut.get(label_value, "Unclassified"), label_value

# Step 3: Create 3D plot with relevant regions and MNI coordinates
def create_combined_3d_plot(relevant_regions_data, region_colors, mni_coords, atlas_data, relevant_region_ids, downsample_factor):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot relevant white matter regions (excluding "Unclassified")
    for i, region_data in enumerate(relevant_regions_data):
        verts, faces, normals, values = measure.marching_cubes(region_data, level=0.5)
        mesh = Poly3DCollection(verts[faces], alpha=0.1, linewidths=0, edgecolors='none')
        mesh.set_facecolor(region_colors[i])
        ax.add_collection3d(mesh)

    # Plot MNI coordinates (including "Unclassified" points)
    colors = plt.cm.jet(np.linspace(0, 1, len(relevant_region_ids)))
    region_colors = {region: colors[i] for i, region in enumerate(relevant_region_ids)}

    for coord in mni_coords:
        region_name, region_idx = identify_region(coord, atlas_data, lut)
        # Plot all MNI coordinates, including "Unclassified"
        ax.scatter(coord[0]//downsample_factor, coord[1]//downsample_factor, coord[2]//downsample_factor, color=region_colors.get(region_idx, 'red'), s=50)

    # Set labels and axis limits
    ax.set_xlim(0, atlas_data.shape[0] // downsample_factor)
    ax.set_ylim(0, atlas_data.shape[1] // downsample_factor)
    ax.set_zlim(0, atlas_data.shape[2] // downsample_factor)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=45, azim=45)

    # Add legend with regions (excluding "Unclassified" from the legend)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=lut[region_id], markersize=10, markerfacecolor=region_colors[region_id]) for region_id in relevant_region_ids if region_id != 0]
    ax.legend(handles=handles, title="Regions", loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.show()

# Step 4: Create a table listing the modeled tracts
def generate_tracts_table(relevant_region_ids, lut):
    # Create a dataframe for displaying the table of modeled tracts
    tracts_info = [(region_id, lut[region_id]) for region_id in relevant_region_ids if region_id != 0]
    df = pd.DataFrame(tracts_info, columns=["Region ID", "White Matter Tract"])
    print("Modeled White Matter Tracts:")
    print(df.to_string(index=False))  # Print the table
    return df

# Main function
def main():
    file_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz'  # Update with your local path
    downsample_factor = 2  # Downsampling factor to reduce memory usage and complexity

    # Example MNI coordinates (replace with actual MNI coordinates)
    mni_coords = [(107, 89, 37), (2, -6, 6)]  # Add actual MNI points

    # Load the atlas data for region identification
    atlas_img = nib.load(file_path)
    atlas_data = atlas_img.get_fdata()

    # Find unique regions corresponding to the MNI coordinates
    relevant_region_ids = set()
    for coord in mni_coords:
        _, region_id = identify_region(coord, atlas_data, lut)
        relevant_region_ids.add(region_id)

    # Load and extract only the relevant regions (excluding "Unclassified" for 3D modeling)
    relevant_regions_data, region_colors, _ = load_relevant_regions(file_path, relevant_region_ids, downsample_factor=downsample_factor)

    # Create the combined plot (plotting MNI coordinates including "Unclassified" but skipping 3D model generation for "Unclassified")
    create_combined_3d_plot(relevant_regions_data, region_colors, mni_coords, atlas_data, relevant_region_ids, downsample_factor)

    # Generate and display the table of modeled tracts (excluding "Unclassified")
    generate_tracts_table(relevant_region_ids, lut)

if __name__ == "__main__":
    main()
