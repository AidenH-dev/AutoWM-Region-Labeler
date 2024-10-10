import pandas as pd
import os
import csv
from collections import defaultdict
import numpy as np

try:
    import cupy as cp  # GPU-accelerated numpy-like library
    USE_GPU = True
    print("GPU acceleration enabled with CuPy.")
except ImportError:
    USE_GPU = False
    print("CuPy not available. Falling back to CPU.")

# Define the input folder, atlas path, and output folder
input_folder = '../AutoWM-Region-Labeler/mm_adjusted_output'
atlas_path = '../AutoWM-Region-Labeler/JHU_atlas/jhu_labels.csv'  # Updated to the new CSV atlas
output_file = '../AutoWM-Region-Labeler/region_statistics_output.csv'

# Load the JHU White Matter Atlas CSV file
print("Loading JHU White Matter Atlas CSV...")
atlas_df = pd.read_csv(atlas_path)  # Reading atlas from CSV

# Extract atlas coordinates in mm and clusters
atlas_coords = atlas_df[['x.mm', 'y.mm', 'z.mm']].values  # Ensure we're using only .mm coordinates
atlas_clusters = atlas_df['cluster'].values

# Look-up table (LUT) for white matter regions.
lut = {
    1: "1 Unclassified",
    2: "2 Middle_cerebellar_peduncle",
    3: "3 Pontine_crossing_tract_(a_part_of_MCP)",
    4: "4 Genu_of_corpus_callosum",
    5: "5 Body_of_corpus_callosum",
    6: "6 Splenium_of_corpus_callosum",
    7: "7 Fornix_(column_and_body_of_fornix)",
    8: "8 Corticospinal_tract_R",
    9: "9 Corticospinal_tract_L",
    10: "10 Medial_lemniscus_R",
    11: "11 Medial_lemniscus_L",
    12: "12 Inferior_cerebellar_peduncle_R",
    13: "13 Inferior_cerebellar_peduncle_L",
    14: "14 Superior_cerebellar_peduncle_R",
    15: "15 Superior_cerebellar_peduncle_L",
    16: "16 Cerebral_peduncle_R",
    17: "17 Cerebral_peduncle_L",
    18: "18 Anterior_limb_of_internal_capsule_R",
    19: "19 Anterior_limb_of_internal_capsule_L",
    20: "20 Posterior_limb_of_internal_capsule_R",
    21: "21 Posterior_limb_of_internal_capsule_L",
    22: "22 Retrolenticular_part_of_internal_capsule_R",
    23: "23 Retrolenticular_part_of_internal_capsule_L",
    24: "24 Anterior_corona_radiata_R",
    25: "25 Anterior_corona_radiata_L",
    26: "26 Superior_corona_radiata_R",
    27: "27 Superior_corona_radiata_L",
    28: "28 Posterior_corona_radiata_R",
    29: "29 Posterior_corona_radiata_L",
    30: "30 Posterior_thalamic_radiation_(include_optic_radiation)_R",
    31: "31 Posterior_thalamic_radiation_(include_optic_radiation)_L",
    32: "32 Sagittal_stratum_(include_inferior_longitidinal_fasciculus_and_inferior_fronto-occipital_fasciculus)_R",
    33: "33 Sagittal_stratum_(include_inferior_longitidinal_fasciculus_and_inferior_fronto-occipital_fasciculus)_L",
    34: "34 External_capsule_R",
    35: "35 External_capsule_L",
    36: "36 Cingulum_(cingulate_gyrus)_R",
    37: "37 Cingulum_(cingulate_gyrus)_L",
    38: "38 Cingulum_(hippocampus)_R",
    39: "39 Cingulum_(hippocampus)_L",
    40: "40 Fornix_(cres)_/_Stria_terminalis_(can_not_be_resolved_with_current_resolution)_R",
    41: "41 Fornix_(cres)_/_Stria_terminalis_(can_not_be_resolved_with_current_resolution)_L",
    42: "42 Superior_longitudinal_fasciculus_R",
    43: "43 Superior_longitudinal_fasciculus_L",
    44: "44 Superior_fronto-occipital_fasciculus_(could_be_a_part_of_anterior_internal_capsule)_R",
    45: "45 Superior_fronto-occipital_fasciculus_(could_be_a_part_of_anterior_internal_capsule)_L",
    46: "46 Uncinate_fasciculus_R",
    47: "47 Uncinate_fasciculus_L",
    48: "48 Tapetum_R",
    49: "49 Tapetum_L",
    50: " 50",
}

# Function to calculate distance using either numpy or cupy (for GPU offloading)
# Function to calculate distance using either numpy or cupy (for GPU offloading)
def calculate_distance(coord, atlas_coords):
    coord = np.array(coord, dtype=np.float64)  # Ensure coord is a numpy array with proper data type
    atlas_coords = np.array(atlas_coords, dtype=np.float64)  # Ensure atlas_coords is also a numpy array
    
    if USE_GPU:
        coord = cp.array(coord)
        atlas_coords = cp.array(atlas_coords)
        distances = cp.sqrt(cp.sum((atlas_coords - coord) ** 2, axis=1))
        return distances
    else:
        distances = np.sqrt(np.sum((atlas_coords - coord) ** 2, axis=1))
        return distances


# Function to identify the region and cluster corresponding to the given voxel coordinates.
def identify_region(voxel_coords, atlas_coords, atlas_clusters, lut, threshold=0.5):
    distances = calculate_distance(voxel_coords, atlas_coords)
    min_dist_idx = np.argmin(distances) if not USE_GPU else cp.argmin(distances)
    min_dist = distances[min_dist_idx]

    if min_dist <= threshold:
        cluster_value = atlas_clusters[min_dist_idx]
        region_name = lut.get(cluster_value, "Unclassified Reg No Label")
        return region_name, cluster_value
    else:
        return "Not found", "No cluster"

# Function to find the closest region for a point that is not found
def find_closest_region(voxel_coords, atlas_coords, atlas_clusters, lut):
    distances = calculate_distance(voxel_coords, atlas_coords)
    min_dist_idx = np.argmin(distances) if not USE_GPU else cp.argmin(distances)
    cluster_value = atlas_clusters[min_dist_idx]
    region_name = lut.get(cluster_value, "Unclassified Reg No Label")
    return region_name, cluster_value

# Updated function to process the MNI coordinates from a CSV file
def process_coordinates(coords, atlas_coords, atlas_clusters, lut, threshold=0.5):
    region_count = defaultdict(lambda: defaultdict(int))
    not_found_test_count = 0

    for coord in coords:
        cluster_number = int(coord[0])  # Get the cluster number from the input
        voxel_coords = coord[1:4]  # Get the voxel coordinates (x, y, z)

        # Process only coordinates labeled "Not found"
        if coord[4].strip().lower() == "not found":  # Ensure case-insensitive check
            region_name, cluster_value = find_closest_region(voxel_coords, atlas_coords, atlas_clusters, lut)
            region_count[cluster_number][f"Closest Region: {region_name}"] += 1
            not_found_test_count += 1
            #print(f"Processed Cluster: {cluster_number}, Count: {not_found_test_count}")
    
    return region_count

# Function to read MNI coordinates and cluster info from a CSV file
def read_mni_coords_from_csv(input_filename):
    df = pd.read_csv(input_filename, header=0)  # Adjust header if needed
    df.columns = df.columns.str.strip()
    
    # Filter rows where "region name" is "Not found"
    not_found_df = df[df['region name'].str.strip().str.lower() == "not found"]
    
    return not_found_df[['cluster', 'x', 'y', 'z', 'region name']].values  # Including the cluster info

# Function to write the statistics to the output CSV file
def write_statistics_to_csv(not_found_region_count, total_points, filename, output_csv):
    headers = ['File', 'Cluster', 'Region', 'Count', 'Percentage']
    with open(output_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if os.stat(output_csv).st_size == 0:
            csvwriter.writerow(headers)
        
        for cluster, region_count in not_found_region_count.items():
            cluster_total_points = sum(region_count.values())
            for region, count in region_count.items():
                percentage = (count / cluster_total_points) * 100 if cluster_total_points > 0 else 0
                csvwriter.writerow([filename, cluster, region, count, f"{percentage:.2f}%"])

# Main function to process all files in the input folder
def process_all_files(input_folder, atlas_coords, atlas_clusters, lut, output_csv):
    total_points = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_folder, filename)

            print(f"Processing file: {filename}")

            # Read MNI coordinates
            mni_coords = read_mni_coords_from_csv(input_filepath)

            # Process the coordinates using identify_region and closest region handling
            cluster_region_count = process_coordinates(mni_coords, atlas_coords, atlas_clusters, lut)

            # Calculate statistics and append to output file
            total_file_points = len(mni_coords)
            write_statistics_to_csv(cluster_region_count, total_file_points, filename, output_csv)

            total_points += total_file_points

# Process all files in the input folder and output statistics to a single CSV file
process_all_files(input_folder, atlas_coords, atlas_clusters, lut, output_file)

print("Program execution completed. The results are stored in the output file.")