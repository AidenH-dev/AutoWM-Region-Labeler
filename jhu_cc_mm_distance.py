import pandas as pd
import os
import csv
from collections import defaultdict
from colorama import Fore, init
import numpy as np

try:
    import cupy as cp  # GPU-accelerated numpy-like library
    USE_GPU = True
    print(Fore.GREEN + "GPU acceleration enabled with CuPy.")
except ImportError:
    USE_GPU = False
    print(Fore.YELLOW + "CuPy not available. Falling back to CPU.")

# Initialize colorama for cross-platform support.
init(autoreset=True)

# Define the input folder, atlas path, and output folder
input_folder_name = 'mm_adjusted'
input_folder = '../mm'
atlas_path = '../AutoWM-Region-Labeler/JHU_atlas/jhu_labels.csv'  # Updated to the new CSV atlas
output_folder = f'../AutoWM-Region-Labeler/{input_folder_name}_output'

# Load the JHU White Matter Atlas CSV file
print(Fore.CYAN + "Loading JHU White Matter Atlas CSV...")
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
def calculate_distance(coord, atlas_coords):
    if USE_GPU:
        coord = cp.array(coord)
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

# Function to read MNI coordinates and cluster info from a CSV file
def read_mni_coords_from_csv(input_filename):
    df = pd.read_csv(input_filename, header=0)  # Adjust header if needed
    df.columns = df.columns.str.strip()
    return df[['cluster', 'x.mm', 'y.mm', 'z.mm']].values  # Including the cluster info

# Main function that processes the input MNI coordinates.
def process_coordinates(coords, atlas_coords, atlas_clusters, lut, threshold=0.5):
    cluster_region_count = defaultdict(lambda: defaultdict(int))
    results = []
    
    for coord in coords:
        cluster_number = int(coord[0])  # Get the cluster number from the input
        voxel_coords = coord[1:]  # Get the voxel coordinates (x, y, z)
        region_name, atlas_cluster = identify_region(voxel_coords, atlas_coords, atlas_clusters, lut, threshold)
        
        results.append([cluster_number, *voxel_coords, region_name])
        cluster_region_count[cluster_number][region_name] += 1
    
    return results, cluster_region_count

# Function to write the processed results to a CSV file.
def write_to_csv(results, output_filename):
    headers = ['cluster', 'x', 'y', 'z', 'region name']
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        for result in results:
            csvwriter.writerow(result)

# Function to calculate and print region statistics for each cluster, and write to a CSV file.
def print_region_statistics(cluster_region_count, total_points, filename, output_csv=None):
    print(Fore.CYAN + f"\nRegion Distribution Statistics for {filename} by Cluster:")
    
    if output_csv:
        headers = ['File', 'Cluster', 'Region', 'Count', 'Percentage']
        with open(output_csv, 'a', newline='') as csvfile:  # Open file in append mode
            csvwriter = csv.writer(csvfile)
            # Write headers only if the file is new or empty
            if os.stat(output_csv).st_size == 0:
                csvwriter.writerow(headers)
            
            for cluster, region_count in cluster_region_count.items():
                print(Fore.YELLOW + f"\nCluster: {cluster}")
                cluster_total_points = sum(region_count.values())
                
                for region, count in region_count.items():
                    percentage = (count / cluster_total_points) * 100
                    print(Fore.GREEN + f"Region: {region}, " + Fore.CYAN + f"Count: {count}, " + Fore.MAGENTA + f"Percentage: {percentage:.2f}%")
                    
                    # Write to CSV
                    csvwriter.writerow([filename, cluster, region, count, f"{percentage:.2f}%"])
    else:
        for cluster, region_count in cluster_region_count.items():
            print(Fore.YELLOW + f"\nCluster: {cluster}")
            cluster_total_points = sum(region_count.values())
            
            for region, count in region_count.items():
                percentage = (count / cluster_total_points) * 100
                print(Fore.GREEN + f"Region: {region}, " + Fore.CYAN + f"Count: {count}, " + Fore.MAGENTA + f"Percentage: {percentage:.2f}%")

# Modify process_all_files to pass an output CSV file for the statistics.
def process_all_files(input_folder, output_folder, threshold=1):
    total_cluster_region_count = defaultdict(lambda: defaultdict(int))
    total_points = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_folder, filename)
            output_filename = filename.replace('.csv', '_output.csv')
            output_filepath = os.path.join(output_folder, output_filename)

            print(Fore.CYAN + f"\nProcessing file: {filename}")

            mni_coords = read_mni_coords_from_csv(input_filepath)
            results, cluster_region_count = process_coordinates(mni_coords, atlas_coords, atlas_clusters, lut, threshold)
            write_to_csv(results, output_filepath)

            total_file_points = len(mni_coords)
            stats_output_filepath = os.path.join(output_folder, 'stats_output.csv')
            print(Fore.GREEN + f"\nOverview for {filename}:")
            print_region_statistics(cluster_region_count, total_file_points, filename, stats_output_filepath)

            total_points += total_file_points
            for cluster, region_count in cluster_region_count.items():
                for region, count in region_count.items():
                    total_cluster_region_count[cluster][region] += count

    print(Fore.CYAN + "\nTotal Overview for All Files:")
    total_stats_output_filepath = os.path.join(output_folder, 'total_stats_output.csv')
    print_region_statistics(total_cluster_region_count, total_points, 'Total', total_stats_output_filepath)

# Process all files in the input folder
process_all_files(input_folder, output_folder)

print(Fore.GREEN + "Program execution completed. All results are available in the output folder.")
