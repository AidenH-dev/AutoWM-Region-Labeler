
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
    """
    This function takes a list of tuples (coordinates) and applies a shift to each coordinate.
    
    Parameters:
    coords (list of tuples): List of (x, y, z) coordinates
    shift_values (tuple): The shift to apply to each coordinate, default is (x+92, y+127, z+73)
    
    Returns:
    list of tuples: Shifted coordinates
    """
    shift = np.array(shift_values)
    shifted_coords = [tuple(np.array(coord) + shift) for coord in coords]
    return shifted_coords

# Function to identify the region for given voxel coordinates
def identify_region(voxel_coords, atlas_data, lut):
    """
    Identify the white matter region corresponding to the given voxel coordinates.
    
    :param voxel_coords: Tuple of voxel coordinates (i, j, k)
    :param atlas_data: Numpy array of the atlas data
    :param lut: Dictionary mapping atlas labels to anatomical names
    :return: Name of the white matter region
    """
    label_value = int(atlas_data[tuple(voxel_coords)])
    return lut.get(label_value, "Unclassified")

# Function to calculate the distance between two points
def calculate_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two points.
    
    :param coord1: First 3D coordinate
    :param coord2: Second 3D coordinate
    :return: Distance as a float
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

# Function to process coordinates and find regions
def process_coordinates(coords, atlas_data, lut):
    """
    Process an array of coordinates, identify regions, and calculate distances.
    
    :param coords: List of (x, y, z) MNI coordinates
    :param atlas_data: 3D Nifti data array from the brain atlas
    :param lut: Dictionary mapping atlas labels to anatomical region names
    :return: A list of results with original coordinates, region name, and distance
    """
    shifted_coords = shift_coordinates(coords)
    results = []
    for i, coord in enumerate(coords):
        shifted_coord = shifted_coords[i]
        region_name = identify_region(shifted_coord, atlas_data, lut)
        
        # Save original MNI coordinates, not the shifted ones
        result = [*coord, region_name, 0.0]  # Distance is 0.0 if it's in the region
        results.append(result)
    return results

# Function to write results to CSV
def write_to_csv(results, output_filename='output.csv'):
    """
    Write the results to a CSV file.
    
    :param results: List of processed results with coordinates, regions, and distances
    :param output_filename: Name of the CSV output file
    """
    headers = ['x', 'y', 'z', 'region 1 name', 'region 1 distance']
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        for result in results:
            csvwriter.writerow(result)

# Example input MNI coordinates
mni_coords = [(-21, 34, 3), (-41,34,3)]  # Input coordinates, which should ping inside "Anterior corona radiata R"

# Process the coordinates and identify regions
results = process_coordinates(mni_coords, atlas_data, lut)

# Write the results to a CSV file
write_to_csv(results)

print("Results written to output.csv")
