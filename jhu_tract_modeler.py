import nibabel as nib
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

    for region_idx, region_name in lut.items():
        if region_idx == 0:  # Skip "Unclassified"
            continue
        
        # Extract region
        region_data = np.zeros_like(nii_data)
        region_data[nii_data == region_idx] = 1

        # If the region is present, add it to the list
        if np.any(region_data):
            all_regions_data.append(region_data)

            # Assign random color to the region
            color = (random.random(), random.random(), random.random())
            all_colors.append(color)

    return all_regions_data, all_colors

# Step 2: Downsample the region data for performance optimization
def downsample(data, factor=2):
    """Downsample the 3D data to reduce complexity."""
    return data[::factor, ::factor, ::factor]

# Step 3: Create 3D plot with all regions
def create_combined_3d_plot(all_regions_data, all_colors):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, region_data in enumerate(all_regions_data):
        # Downsample the region data for performance improvement
        region_data = downsample(region_data, factor=2)

        # Use marching cubes to generate 3D surface for each region
        verts, faces, normals, values = measure.marching_cubes(region_data, level=0.5)

        # Create a 3D mesh using the vertices and faces from marching cubes
        mesh = Poly3DCollection(verts[faces], alpha=0.1, linewidths=0, edgecolors='none')  # Increased transparency and removed edge rendering
        mesh.set_facecolor(all_colors[i])

        ax.add_collection3d(mesh)

    # Set axis limits based on the first region's size
    ax.set_xlim(0, verts[:, 0].max())
    ax.set_ylim(0, verts[:, 1].max())
    ax.set_zlim(0, verts[:, 2].max())

    # Set labels and view angle
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=45, azim=45)

    plt.show()

# Main function to load regions and generate the 3D plot
def main():
    file_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz'  # Update with your local path
    all_regions_data, all_colors = load_and_extract_regions(file_path)
    create_combined_3d_plot(all_regions_data, all_colors)

if __name__ == "__main__":
    main()
