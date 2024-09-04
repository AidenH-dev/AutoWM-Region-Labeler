import nibabel as nib
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the JHU White Matter Atlas
atlas_img = nib.load('../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz')
atlas_data = atlas_img.get_fdata()

# Manually create the Look-Up Table (LUT) based on the provided file
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

def identify_region(voxel_coords, atlas_data, lut):
    """
    Identify the white matter region corresponding to the given voxel coordinates.
    
    :param voxel_coords: Tuple of voxel coordinates (i, j, k)
    :param atlas_data: Numpy array of the atlas data
    :param lut: Dictionary mapping atlas labels to anatomical names
    :return: Name of the white matter region
    """
    # Check if each voxel coordinate is within bounds
    if any(v < 0 or v >= atlas_data.shape[i] for i, v in enumerate(voxel_coords)):
        return "Coordinates out of bounds"
    
    label_value = int(atlas_data[voxel_coords])
    
    if label_value in lut:
        return lut[label_value]
    else:
        return "Region not found"

def check_cluster_region(cluster_coords, atlas_data, lut, distance_threshold=5):
    """
    Check if a cluster of points are within the same region.
    
    :param cluster_coords: List of voxel coordinates [(i, j, k), ...]
    :param atlas_data: Numpy array of the atlas data
    :param lut: Dictionary mapping atlas labels to anatomical names
    :param distance_threshold: Maximum distance between points to consider them in the same cluster
    :return: Dictionary with regions and number of points in each region
    """
    # Calculate pairwise distances
    distances = squareform(pdist(cluster_coords))
    
    # Check if all points are within the distance threshold
    cluster_regions = {}
    for idx, voxel_coords in enumerate(cluster_coords):
        region_name = identify_region(voxel_coords, atlas_data, lut)
        
        if region_name not in cluster_regions:
            cluster_regions[region_name] = 0
        cluster_regions[region_name] += 1
    
    # Check if points are within the same region
    all_same_region = all(value == cluster_regions[region_name] for region_name, value in cluster_regions.items())
    
    return cluster_regions, all_same_region

# Example: Cluster of MNI coordinates
cluster_coords = [
    (69, 161, 73), (70, 161, 74), (68, 160, 72), (71, 162, 75),
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
    (98, 189, 102), (41, 133, 45), (99, 190, 103), (81, 154, 73)
]
cluster_regions, all_same_region = check_cluster_region(cluster_coords, atlas_data, lut)

print("Cluster Regions:", cluster_regions)
print("All Points in Same Region:", all_same_region)

# Visualization
def plot_cluster_regions(cluster_coords, cluster_regions, lut):
    """
    Plot the cluster regions in a 3D scatter plot.
    
    :param cluster_coords: List of voxel coordinates [(i, j, k), ...]
    :param cluster_regions: Dictionary with regions and number of points in each region
    :param lut: Dictionary mapping atlas labels to anatomical names
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Assign a unique color to each region
    unique_regions = list(cluster_regions.keys())
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_regions)))
    region_colors = {region: colors[i] for i, region in enumerate(unique_regions)}
    
    # Plot each point with the color corresponding to its region
    for idx, voxel_coords in enumerate(cluster_coords):
        region_name = identify_region(voxel_coords, atlas_data, lut)
        ax.scatter(voxel_coords[0], voxel_coords[1], voxel_coords[2], color=region_colors[region_name], label=region_name)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cluster Regions Visualization')
    
    # Create a legend with unique regions
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=region_name, markersize=10, markerfacecolor=region_colors[region_name]) for region_name in unique_regions]
    ax.legend(handles=handles, title="Regions", loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.show()

# Plot the results
plot_cluster_regions(cluster_coords, cluster_regions, lut)
