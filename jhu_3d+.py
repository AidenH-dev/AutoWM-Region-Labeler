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

# Pre Step 1: Prep MNI coords to read as MRIcron Coordinates
def shift_coordinates(coords, shift_values=(92, 127, 73)):
    shift = np.array(shift_values)
    shifted_coords = [tuple(np.array(coord) + shift) for coord in coords]
    return shifted_coords

# Step 1: Load and extract all regions
def load_all_regions(file_path, downsample_factor=2):
    nii = nib.load(file_path)
    nii_data = nii.get_fdata()

    all_regions_data = []
    all_colors = []

    for region_idx in lut.keys():
        region_data = np.zeros_like(nii_data)
        region_data[nii_data == region_idx] = 1

        if np.any(region_data):
            region_data = region_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            all_regions_data.append(region_data)
            color = (random.random(), random.random(), random.random())
            all_colors.append(color)

    return all_regions_data, all_colors, nii_data

# Step 2: Identify the white matter region corresponding to the coordinates
def identify_region(voxel_coords, atlas_data, lut):
    label_value = int(atlas_data[tuple(voxel_coords)])
    return lut.get(label_value, "Unclassified"), label_value

# Step 3: Create 3D plot with all regions
def create_combined_3d_plot_with_vectors(all_regions_data, region_colors, mni_coords, vectors, atlas_data, downsample_factor):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all white matter regions
    for i, region_data in enumerate(all_regions_data):
        verts, faces, normals, values = measure.marching_cubes(region_data, level=0.5)
        mesh = Poly3DCollection(verts[faces], alpha=0.1, linewidths=0, edgecolors='none')
        mesh.set_facecolor(region_colors[i])
        ax.add_collection3d(mesh)

    # Plot MNI coordinates and vectors
    colors = plt.cm.jet(np.linspace(0, 1, len(lut.keys())))
    region_colors = {region: colors[i] for i, region in enumerate(lut.keys())}

    for i, coord in enumerate(mni_coords):
        region_name, region_idx = identify_region(coord, atlas_data, lut)
        downsampled_coord = [coord[i] // downsample_factor for i in range(3)]

        # Plot the MNI coordinate
        ax.scatter(*downsampled_coord, color=region_colors.get(region_idx, 'red'), s=50)

        # Plot vectors to the closest regions
        for vec in vectors[i]:
            vector = np.array(vec)
            endpoint = downsampled_coord + vector
            ax.plot([downsampled_coord[0], endpoint[0]],
                    [downsampled_coord[1], endpoint[1]],
                    [downsampled_coord[2], endpoint[2]],
                    color='green')

    ax.set_xlim(0, atlas_data.shape[0] // downsample_factor)
    ax.set_ylim(0, atlas_data.shape[1] // downsample_factor)
    ax.set_zlim(0, atlas_data.shape[2] // downsample_factor)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=45, azim=45)
    plt.show()

# Main function
def main():
    file_path = '../AutoWM-Region-Labeler/output.csv'  # Path to the CSV file
    atlas_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz'  # Path to the NIfTI file

    # Load the data from the CSV
    data = pd.read_csv(file_path)

    # Extract MNI coordinates
    mni_coords = data[['x', 'y', 'z']].values

    # Extract vectors to the closest regions
    vectors = []
    for i in range(len(data)):
        vectors.append([eval(data.iloc[i]['region 1 vector']), eval(data.iloc[i]['region 2 vector']), eval(data.iloc[i]['region 3 vector'])])

    # Processing MNI coordinates to MRIcron coordinates 
    shifted_coordinates = shift_coordinates(mni_coords)

    # Load all regions
    all_regions_data, region_colors, atlas_data = load_all_regions(atlas_path, downsample_factor=5)

    # Create the combined plot with vectors
    create_combined_3d_plot_with_vectors(all_regions_data, region_colors, shifted_coordinates, vectors, atlas_data, downsample_factor=5)

if __name__ == "__main__":
    main()
