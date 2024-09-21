import nibabel as nib  # Importing the nibabel library for neuroimaging data (e.g., NIfTI files)
import numpy as np  # Importing NumPy for numerical operations
import csv  # Importing the CSV module to write the results to a CSV file
import os  # Importing os to handle directory operations
import pandas as pd  # Importing pandas for reading Excel files
from collections import defaultdict
from colorama import Fore, init

# Initialize colorama for cross-platform support.
init(autoreset=True)

# Define the input folder, atlas path, and output folder
input_folder = '../AutoWM-Region-Labeler/WM_cluster_coords--Active_Passive'
atlas_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-2mm.nii.gz'
output_folder = '../AutoWM-Region-Labeler/Output_WM_cluster_coords--Active_Passive'

# Load the JHU White Matter Atlas NIfTI file from the local folder.
print(Fore.CYAN + "Loading JHU White Matter Atlas...")
atlas_img = nib.load(atlas_path)
print(Fore.GREEN + "Atlas loaded successfully!")

# Extract the actual image data from the loaded NIfTI file.
print(Fore.CYAN + "Extracting atlas data...")
atlas_data = atlas_img.get_fdata()
print(Fore.GREEN + "Atlas data extraction complete.")

# Look-up table (LUT) for white matter regions.
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

# Function to identify the region corresponding to the given voxel coordinates.
def identify_region(voxel_coords, atlas_data, lut):
    if (0 <= voxel_coords[0] < atlas_data.shape[0] and
        0 <= voxel_coords[1] < atlas_data.shape[1] and
        0 <= voxel_coords[2] < atlas_data.shape[2]):
        label_value = int(atlas_data[tuple(voxel_coords)])
        return lut.get(label_value, "Unclassified")
    else:
        return "Out of Bounds"

# Function to read MNI coordinates from an Excel file.
def read_mni_coords_from_excel(input_filename):
    df = pd.read_excel(input_filename)
    return list(df[['x', 'y', 'z']].to_records(index=False))

# Main function that processes the input MNI coordinates.
def process_coordinates(coords, atlas_data, lut):
    region_count = defaultdict(int)
    results = []
    for coord in coords:
        region_name = identify_region(coord, atlas_data, lut)
        results.append([*coord, region_name])
        region_count[region_name] += 1
    return results, region_count

# Function to write the processed results to a CSV file.
def write_to_csv(results, output_filename):
    headers = ['x', 'y', 'z', 'region name']
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        for result in results:
            csvwriter.writerow(result)

# Function to calculate and print region statistics.
def print_region_statistics(region_count, total_points):
    print(Fore.CYAN + "\nRegion Distribution Statistics:")
    for region, count in region_count.items():
        percentage = (count / total_points) * 100
        print(Fore.YELLOW + f"Region: {region}, " + Fore.GREEN + f"Count: {count}, " + Fore.MAGENTA + f"Percentage: {percentage:.2f}%")

# Function to process all files in a folder
def process_all_files(input_folder, output_folder):
    total_region_count = defaultdict(int)
    total_points = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".xlsx"):
            input_filepath = os.path.join(input_folder, filename)
            output_filename = filename.replace('.xlsx', '_output.csv')
            output_filepath = os.path.join(output_folder, output_filename)

            print(Fore.CYAN + f"\nProcessing file: {filename}")

            mni_coords = read_mni_coords_from_excel(input_filepath)
            results, region_count = process_coordinates(mni_coords, atlas_data, lut)
            write_to_csv(results, output_filepath)

            total_file_points = len(mni_coords)
            print(Fore.GREEN + f"\nOverview for {filename}:")
            print_region_statistics(region_count, total_file_points)

            total_points += total_file_points
            for region, count in region_count.items():
                total_region_count[region] += count

    print(Fore.CYAN + "\nTotal Overview for All Files:")
    print_region_statistics(total_region_count, total_points)

# Process all Excel files in the input folder
process_all_files(input_folder, output_folder)

print(Fore.GREEN + "Program execution completed. All results are available in the output folder.")
