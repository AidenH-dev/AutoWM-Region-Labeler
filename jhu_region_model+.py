import nibabel as nib
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

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

    # Plot relevant white matter regions
    for i, region_data in enumerate(relevant_regions_data):
        verts, faces, normals, values = measure.marching_cubes(region_data, level=0.5)
        mesh = Poly3DCollection(verts[faces], alpha=0.1, linewidths=0, edgecolors='none')
        mesh.set_facecolor(region_colors[i])
        ax.add_collection3d(mesh)

    # Plot MNI coordinates
    colors = plt.cm.jet(np.linspace(0, 1, len(relevant_region_ids)))
    region_colors = {region: colors[i] for i, region in enumerate(relevant_region_ids)}

    for coord in mni_coords:
        region_name, region_idx = identify_region(coord, atlas_data, lut)
        ax.scatter(coord[0]//downsample_factor, coord[1]//downsample_factor, coord[2]//downsample_factor, color=region_colors[region_idx], s=50)

    # Set labels and axis limits
    ax.set_xlim(0, atlas_data.shape[0] // downsample_factor)
    ax.set_ylim(0, atlas_data.shape[1] // downsample_factor)
    ax.set_zlim(0, atlas_data.shape[2] // downsample_factor)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=45, azim=45)

    # Add legend with regions
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=lut[region_id], markersize=10, markerfacecolor=region_colors[region_id]) for region_id in relevant_region_ids]
    ax.legend(handles=handles, title="Regions", loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.show()

# Main function
def main():
    file_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz'  # Update with your local path
    downsample_factor = 2  # Downsampling factor to reduce memory usage and complexity

    # Example MNI coordinates (replace with actual MNI coordinates)
    mni_coords = [(69, 161, 73), (70, 161, 74), (68, 160, 72), (71, 162, 75),
    (72, 163, 76), (67, 159, 71), (73, 164, 77), (66, 158, 70),
    (74, 165, 78), (65, 157, 69), (75, 166, 79), (64, 156, 68),
    (76, 167, 80), (63, 155, 67), (77, 168, 81), (62, 154, 66),
    (78, 169, 82), (61, 153, 65), (79, 170, 83), (60, 152, 64),
    (80, 171, 84), (59, 151, 63), (81, 172, 85), (58, 150, 62),
    (82, 173, 86), (57, 149, 61), (83, 174, 87), (56, 148, 60),
    (84, 175, 88), (55, 147, 59), (85, 176, 89), (54, 146, 58),
    (86, 177, 90), (53, 145, 57), (87, 178, 91), (52, 144, 56),
    (88, 179, 92), (51, 143, 55), (89, 180, 93), (50, 142, 54),
    (90, 181, 94), (49, 141, 53), (91, 182, 95), (48, 140, 52),
    (92, 183, 96), (47, 139, 51), (93, 184, 97), (46, 138, 50),
    (94, 185, 98), (45, 137, 49), (95, 186, 99), (44, 136, 48),
    (96, 187, 100), (43, 135, 47), (97, 188, 101), (42, 134, 46),
    (98, 189, 102), (41, 133, 45), (99, 190, 103), (81, 154, 73)]  # Add actual MNI points

    # Load the atlas data for region identification
    atlas_img = nib.load(file_path)
    atlas_data = atlas_img.get_fdata()

    # Find unique regions corresponding to the MNI coordinates
    relevant_region_ids = set()
    for coord in mni_coords:
        _, region_id = identify_region(coord, atlas_data, lut)
        relevant_region_ids.add(region_id)

    # Load and extract only the relevant regions, with downsampling for optimization
    relevant_regions_data, region_colors, _ = load_relevant_regions(file_path, relevant_region_ids, downsample_factor=downsample_factor)

    # Create the combined plot
    create_combined_3d_plot(relevant_regions_data, region_colors, mni_coords, atlas_data, relevant_region_ids, downsample_factor)

if __name__ == "__main__":
    main()
