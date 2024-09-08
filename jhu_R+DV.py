
# Program Description:
# This program processes an array of 3D MNI coordinates, applies a predefined shift to convert them into MRIcron coordinates, 
# and then identifies the closest white matter brain regions using a JHU White Matter Atlas. The program works as follows:
# 
# 1. **Coordinate Shifting**: The coordinates provided as input are shifted by default values (x+92, y+127, z+73), which is 
#    necessary to align the input coordinates with the coordinate system used by MRIcron.
#    
# 2. **Region Identification**: For each shifted coordinate, the program uses an atlas of labeled brain regions to find the 
#    closest three regions. These regions are determined by calculating the Euclidean distance between the shifted coordinates 
#    and the coordinates of all voxels in the atlas. For each input coordinate, the three closest regions and their respective 
#    distances are recorded.
# 
# 3. **Output Format**: The program outputs the results into a CSV file. Each row of the CSV contains the following columns:
#    - The x, y, and z coordinates (after shifting)
#    - The names of the three closest brain regions
#    - The distances to these three closest brain regions
#    
# This implementation uses the JHU White Matter Atlas for region identification and assumes the atlas is pre-loaded with 
# voxel data. It also assumes that the input coordinates are valid MNI coordinates. The output CSV file can be used for 
# further analysis or reporting.

import nibabel as nib
import numpy as np
import csv

# Load the JHU White Matter Atlas
atlas_img = nib.load('../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz')
atlas_data = atlas_img.get_fdata()

# Look-up table (LUT) for white matter regions
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

# Function to shift coordinates
def shift_coordinates(coords, shift_values=(92, 127, 73)):
    shift = np.array(shift_values)
    shifted_coords = [tuple(np.array(coord) + shift) for coord in coords]
    return shifted_coords

# Function to identify the region for given voxel coordinates
def identify_region(voxel_coords, atlas_data, lut):
    label_value = int(atlas_data[tuple(voxel_coords)])
    return lut.get(label_value, "Unclassified")

# Function to calculate the Euclidean distance between two points
def calculate_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

# Function to process coordinates and find regions with distances to the next 3 closest regions
def process_coordinates(coords, atlas_data, lut):
    shifted_coords = shift_coordinates(coords)
    results = []
    
    atlas_shape = atlas_data.shape
    all_voxels = np.array(np.where(atlas_data != 0)).T  # All non-zero voxels
    
    for i, coord in enumerate(coords):
        shifted_coord = shifted_coords[i]
        
        # Check if the coordinate falls inside a region
        region_name = identify_region(shifted_coord, atlas_data, lut)
        region_distances = {}
        
        if region_name != "Unclassified":
            # If it falls inside a region, set that as region 1 and distance as 0
            result = [*coord, region_name, 0.0]
        else:
            # If unclassified, calculate the distances to other regions
            result = [*coord, "Unclassified", 0.0]
        
        # Iterate through all voxels in the atlas to calculate distances
        for voxel in all_voxels:
            current_region = identify_region(voxel, atlas_data, lut)
            if current_region != "Unclassified" and current_region != region_name:  # Skip unclassified and region 1
                distance = calculate_distance(shifted_coord, voxel)
                if current_region in region_distances:
                    region_distances[current_region].append(distance)
                else:
                    region_distances[current_region] = [distance]
        
        # For each region, use the minimum distance (closest voxel)
        min_distances = {region: min(distances) for region, distances in region_distances.items()}
        
        # Sort regions by distance
        sorted_regions = sorted(min_distances.items(), key=lambda item: item[1])
        
        # Add the next 3 closest regions to the result
        for idx in range(3):
            if idx < len(sorted_regions):
                region_name, distance = sorted_regions[idx]
                result.append(region_name)
                result.append(distance)
            else:
                result.append("None")  # If there are fewer than 3 regions
                result.append("N/A")
        
        results.append(result)
    
    return results

# Function to write results to CSV
def write_to_csv(results, output_filename='output.csv'):
    headers = ['x', 'y', 'z', 'region 1 name', 'region 1 distance',
               'region 2 name', 'region 2 distance',
               'region 3 name', 'region 3 distance',
               'region 4 name', 'region 4 distance']
    
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        for result in results:
            csvwriter.writerow(result)

# Example input MNI coordinates
mni_coords = [(12, -91, 11), (27, -88, 20), (15, -86, 5), (-6, -94, 11),    (-8, -90, -1), (-14, -91, 24), (-24, -32, 2), (-9, -28, -2),    (-20, -14, 17), (2, -40, -42), (-6, -40, -44), (26, -61, 56),    (18, -54, 65), (27, -49, 56), (-20, -62, 54), (-20, -61, -26),    (-15, -67, -30), (-15, -73, -36), (21, -61, -26), (28, -52, -31),    (42, -8, 40), (34, -2, 44), (18, -24, 60), (21, 2, 53),    (12, 8, 50), (28, 23, 11), (28, 32, 5), (20, 53, -7),    (-51, 5, -10), (-48, -6, -12), (-32, -1, 46), (56, -18, 0),    (15, -8, 14), (12, -18, 8), (18, 0, 18), (-16, 17, 38),    (-10, 14, 46), (21, 12, 38)]


# Process the coordinates and identify regions
results = process_coordinates(mni_coords, atlas_data, lut)

# Write the results to a CSV file
write_to_csv(results)

print("Results written to output.csv")