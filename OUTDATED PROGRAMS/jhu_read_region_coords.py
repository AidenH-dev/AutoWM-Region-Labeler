import nibabel as nib
import numpy as np
from scipy.spatial import distance

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

def get_region_coordinates(atlas_data, lut):
    """
    Output every coordinate within each region.
    
    :param atlas_data: Numpy array of the atlas data
    :param lut: Dictionary mapping atlas labels to anatomical names
    :return: Dictionary with region names as keys and lists of coordinates as values
    """
    region_coordinates = {region: [] for region in lut.values() if region != "Unclassified"}
    
    # Iterate through each voxel in the atlas
    for i in range(atlas_data.shape[0]):
        for j in range(atlas_data.shape[1]):
            for k in range(atlas_data.shape[2]):
                label_value = int(atlas_data[i, j, k])
                if label_value in lut:
                    region_name = lut[label_value]
                    if region_name != "Unclassified":
                        region_coordinates[region_name].append((i, j, k))
    
    return region_coordinates

def find_closest_regions(unclassified_coord, region_coords, k=3):
    """
    Find the closest regions to a given unclassified coordinate.
    
    :param unclassified_coord: Coordinate in unclassified region
    :param region_coords: Dictionary with region names as keys and lists of coordinates as values
    :param k: Number of closest regions to find
    :return: List of closest regions with distances
    """
    distances = []
    
    for region_name, coords in region_coords.items():
        if len(coords) > 0:
            # Compute distances from the unclassified coordinate to all coordinates in this region
            region_distances = [distance.euclidean(unclassified_coord, coord) for coord in coords]
            min_distance = min(region_distances)
            distances.append((region_name, min_distance))
    
    # Sort regions by distance and get the closest k regions
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def main(input_coord):
    # Get the coordinates for each region, excluding unclassified
    region_coordinates = get_region_coordinates(atlas_data, lut)
    
    # Check if the input coordinate is in an unclassified region
    x, y, z = input_coord
    if x < 0 or x >= atlas_data.shape[0] or y < 0 or y >= atlas_data.shape[1] or z < 0 or z >= atlas_data.shape[2]:
        print("Coordinates out of bounds")
        return

    if int(atlas_data[x, y, z]) == 0:  # Check if it is in the "Unclassified" region
        print(f"Input Coordinate {input_coord} is in an unclassified region.")
        closest_regions = find_closest_regions(input_coord, region_coordinates)
        
        # Output the closest regions
        print(f"Closest Regions to {input_coord}:")
        for region_name, dist in closest_regions:
            print(f"Region: {region_name}, Distance: {dist:.2f}")
    else:
        print(f"Input Coordinate {input_coord} is not in an unclassified region.")

# Example usage
input_coord = (57, 161, 74)  # Replace with your input coordinate
main(input_coord)
