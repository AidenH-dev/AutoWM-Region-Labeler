
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

import nibabel as nib  # Importing the nibabel library, which is used for reading and working with neuroimaging data (e.g., NIfTI files).
import numpy as np  # Importing NumPy for numerical operations, such as working with arrays.
import csv  # Importing the CSV module to write the results to a CSV file.

# Load the JHU White Matter Atlas NIfTI file.
# This file contains labeled regions of white matter in the brain, stored as a 3D image.
# Each voxel in the 3D image has an integer label that corresponds to a specific white matter region.
atlas_img = nib.load('../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-2mm.nii.gz')

# Extract the actual image data from the loaded NIfTI file.
# This data is stored as a NumPy array, where each element in the array represents a voxel in the brain.
# The value of each voxel is an integer that corresponds to a specific region in the atlas.
atlas_data = atlas_img.get_fdata()

# Look-up table (LUT) for white matter regions.
# This dictionary maps integer values (from the atlas data) to the corresponding region names.
# For example, if a voxel has a value of 1 in the atlas, it corresponds to the "Middle_cerebellar_peduncle" region.
# The value 0 is used for unclassified regions.
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

# Function to shift the input coordinates.
# MNI coordinates typically need to be shifted to match the coordinate system used by the atlas.
# The shift_values parameter represents the amount by which the MNI coordinates should be shifted.
# The function applies this shift to each coordinate.
def mni_to_mricron(coords, mni_voxel_size=1.5):
    """
    Converts a list of MNI coordinates to MRIcron coordinates, handling different MNI voxel sizes
    and adjusting the rounding method to match MRIcron's output.
    
    Parameters:
        coords (list of tuples): List of MNI coordinates.
        mni_voxel_size (float): Voxel size of the MNI coordinates.
        
    Returns:
        list of tuples: List of MRIcron coordinates.
    """
    scaling_factor = 2.0 / mni_voxel_size  # Adjust for different voxel sizes
    converted_coords = []
    for coord in coords:
        X, Y, Z = coord
        # Scale MNI coordinates to 2mm space if needed
        X_scaled = X * scaling_factor
        Y_scaled = Y * scaling_factor
        Z_scaled = Z * scaling_factor
        # Apply the inverted transformation equations with corrected rounding
        x = int(np.floor((X_scaled + 92) / 2))
        y = int(np.floor((Y_scaled + 128 ) / 2))
        z = int(np.ceil((Z_scaled + 74) / 2))
        converted_coords.append((x, y, z))
    return converted_coords


# Function to identify the region corresponding to the given voxel coordinates.
# The function looks up the voxel value in the atlas (i.e., the white matter region label).
# It then uses the look-up table (LUT) to convert the voxel value into a human-readable region name.
# If the voxel value isn't found in the LUT, the region is marked as "Unclassified."
def identify_region(voxel_coords, atlas_data, lut):
    label_value = int(atlas_data[tuple(voxel_coords)])  # Get the value of the voxel at the given coordinates.
    return lut.get(label_value, "Unclassified")  # Look up the region name using the LUT.

# Function to calculate the Euclidean distance between two 3D coordinates (in voxel space).
# This is used to measure the distance between the current voxel and other nearby regions.
def calculate_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))  # Calculate the Euclidean distance.

# Function to calculate the 3D vector between two points.
# This vector represents the direction and magnitude from one point (voxel) to another.
def calculate_vector(coord1, coord2):
    return np.array(coord2) - np.array(coord1)  # Compute the vector difference between two points.
# Modify the vector formatting function to use square brackets and remove extra spaces
def format_vector(vector):
    # Convert the NumPy array to a string, remove extra spaces, and replace parentheses with square brackets
    return str(vector.tolist()).replace(" ", "")

# Main function that processes the input MNI coordinates.
# For each input coordinate, the function shifts the coordinate to match the atlas space,
# identifies the region it belongs to, and finds the 3 closest regions (if the input is unclassified).
def process_coordinates(coords, atlas_data, lut):
    shifted_coords = mni_to_mricron(coords)  # First, shift the input coordinates.
    results = []  # Initialize an empty list to store the results.
    
    # Get the shape of the atlas data to use for bounds checking.
    atlas_shape = atlas_data.shape
    
    # Find all the voxels in the atlas that have a non-zero value (i.e., all classified voxels).
    all_voxels = np.array(np.where(atlas_data != 0)).T  # Transpose to get coordinates in (x, y, z) format.
    
    # Iterate over the list of input coordinates.
    for i, coord in enumerate(coords):
        shifted_coord = shifted_coords[i]  # Get the shifted coordinate for the current point.
        
        # Check if the coordinate falls inside a specific region in the atlas.
        region_name = identify_region(shifted_coord, atlas_data, lut)
        
        # Initialize dictionaries to store distances and vectors to nearby regions.
        region_distances = {}
        region_vectors = {}
        
        # Initialize the result for this coordinate.
        # If the coordinate is classified, the distance to that region is 0, and the vector is [0, 0, 0].
        if region_name != "Unclassified":
            result = [*coord, region_name, 0.0, "[0,0,0]"]  # Store the classified region info with formatted vector.
        else:
            result = [*coord, "Unclassified", 0.0, "[0,0,0]"]  # Mark as unclassified.
        
        # Loop through all classified voxels in the atlas to find the 3 closest regions.
        for voxel in all_voxels:
            current_region = identify_region(voxel, atlas_data, lut)  # Get the region name of the current voxel.
            
            # Skip unclassified regions and the region we are currently in.
            if current_region != "Unclassified" and current_region != region_name:
                distance = calculate_distance(shifted_coord, voxel)  # Calculate the distance to this voxel.
                vector = calculate_vector(shifted_coord, voxel)  # Calculate the vector to this voxel.
                
                # If this region has already been encountered, keep only the closest voxel.
                if current_region in region_distances:
                    if distance < region_distances[current_region]:
                        region_distances[current_region] = distance
                        region_vectors[current_region] = format_vector(vector)  # Use formatted vector.
                else:
                    # Store the distance and vector for this region.
                    region_distances[current_region] = distance
                    region_vectors[current_region] = format_vector(vector)  # Use formatted vector.
        
        # Sort the regions by distance (ascending order).
        sorted_regions = sorted(region_distances.items(), key=lambda item: item[1])
        
        # Add the next 3 closest regions (by distance) to the result.
        for idx in range(3):
            if idx < len(sorted_regions):
                region_name, distance = sorted_regions[idx]
                vector = region_vectors[region_name]
                result.append(region_name)
                result.append(distance)
                result.append(vector)
            else:
                result.append("None")  # If there are fewer than 3 nearby regions, append "None".
                result.append("N/A")  # Append N/A for missing distance and vector.
                result.append("[N/A]")
        
        # Append the result for this coordinate to the list of results.
        results.append(result)
    
    return results  # Return the processed results.


# Function to write the processed results to a CSV file.
# This allows us to save the results for later use or analysis.
def write_to_csv(results, output_filename='output.csv'):
    # Define the headers for the CSV file.
    headers = ['x', 'y', 'z', 'region 1 name', 'region 1 distance', 'region 1 vector',
               'region 2 name', 'region 2 distance', 'region 2 vector',
               'region 3 name', 'region 3 distance', 'region 3 vector',
               'region 4 name', 'region 4 distance', 'region 4 vector']
    
    # Open a new CSV file in write mode.
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)  # Create a CSV writer object.
        csvwriter.writerow(headers)  # Write the header row.
        
        # Write each result to the CSV file.
        for result in results:
            csvwriter.writerow(result)

# Example input MNI coordinates.
# These are 3D coordinates representing specific locations in the brain, often used in neuroimaging studies.
mni_coords = [(14, -10, -10), (34, -76, 17), (-21, -13, 47), (-2, -40, -43), (38, 0, -18),
 (14, -88, 20), (-26, -88, 16), (-22, -20, 11), (-10, -92, 10), (14, -8, -10),
 (-21, -13, 47), (-14, -13, -10), (14, -10, -10), (34, -76, 17), (-21, -13, 47),
 (-2, -40, -43), (-16, -52, 16), (14, -88, 22), (-22, -76, 32), (38, 0, -18),
 (-22, -20, 11), (-21, -13, 47), (14, -10, -10), (-14, -13, -10), (-28, 2, 36),
 (12, -91, 11), (-8, -94, 12), (-24, -31, 2), (22, -30, 2), (2, -40, -42),
 (-20, -60, -26), (20, -2, 54), (21, -61, -26), (30, 30, 6), (-6, -94, 11),
 (12, -91, 11), (2, -40, -40), (24, -62, 56), (-24, -32, 4), (-16, -61, -25),
 (24, -60, -30), (-15, 16, 38), (24, -30, 0), (15, 16, 38), (32, -2, 46),
 (21, 4, 53), (18, 0, 18), (56, -18, 0), (28, 23, 11), (-32, -1, 46),
 (26, 29, 0), (12, -91, 11), (-6, -94, 11), (-24, -32, 2), (2, -40, -42),
 (26, -61, 56), (-20, -62, 54), (-20, -61, -26), (21, -61, -26), (42, -8, 40),
 (18, -24, 60), (21, 2, 53), (28, 23, 11), (-51, 5, -10), (-32, -1, 46),
 (56, -18, 0), (15, -8, 14), (-16, 17, 38), (21, 12, 38)]

# Process the MNI coordinates and identify regions with distances and vectors.
results = process_coordinates(mni_coords, atlas_data, lut)

# Write the processed results to a CSV file.
write_to_csv(results)

# Print a message indicating that the results have been written to a file.
print("Results written to output.csv")
