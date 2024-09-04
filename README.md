# AutoWM-Region-Labeler-Cluster
Automate the region labeling for characterizing and comparing brain activations in white matter and gray matter during different cognitive states.

# README

## Overview

This script provides tools for identifying and analyzing white matter regions in the brain using the JHU White Matter Atlas. The script loads the JHU White Matter Atlas, defines a Look-Up Table (LUT) mapping atlas labels to anatomical names, and provides functions for identifying specific white matter regions based on voxel coordinates. Additionally, the script can check whether a cluster of points belongs to the same white matter region and visualize the results in 3D.

## Prerequisites

Before running the script, ensure you have the following Python packages installed:

- `nibabel`: For loading neuroimaging data in NIfTI format.
- `numpy`: For numerical operations.
- `scipy`: For spatial distance calculations.
- `matplotlib`: For generating visualizations.

You can install these packages using pip:

```bash
pip install nibabel numpy scipy matplotlib
```

## Files

- `JHU-WhiteMatter-labels-1mm.nii.gz`: The NIfTI file containing the JHU White Matter Atlas.
- `jhu_atlas_lookup_cluster.py`: The main script that includes examples and visualizations.

## Usage

### 1. Load the JHU White Matter Atlas

The script begins by loading the JHU White Matter Atlas using the `nibabel` library:

```python
import nibabel as nib

atlas_img = nib.load('../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz')
atlas_data = atlas_img.get_fdata()
```

### 2. Define the Look-Up Table (LUT)

The LUT is manually created to map the atlas labels to corresponding anatomical names:

```python
lut = {
    0: "Unclassified",
    1: "Middle_cerebellar_peduncle",
    ...
    98: "Region_X",
    99: "Region_Y"
}
```

### 3. Identify a White Matter Region

To identify the white matter region corresponding to specific voxel coordinates, use the `identify_region` function:

```python
def identify_region(voxel_coords, atlas_data, lut):
    # Function implementation
```

Example usage:

```python
region_name = identify_region((69, 161, 73), atlas_data, lut)
print(region_name)
```

### 4. Check Cluster Regions

To check whether a cluster of points belongs to the same white matter region, use the `check_cluster_region` function:

```python
def check_cluster_region(cluster_coords, atlas_data, lut, distance_threshold=5):
    # Function implementation
```

Example usage:

```python
cluster_coords = [(69, 161, 73), (70, 161, 74), (68, 160, 72)]
cluster_regions, all_same_region = check_cluster_region(cluster_coords, atlas_data, lut)
print("Cluster Regions:", cluster_regions)
print("All Points in Same Region:", all_same_region)
```

### 5. Visualize Cluster Regions

The script includes a function to visualize the identified regions in 3D:

```python
def plot_cluster_regions(cluster_coords, cluster_regions, lut):
    # Function implementation
```

Example usage:

```python
plot_cluster_regions(cluster_coords, cluster_regions, lut)
```

### 6. Example

An example is provided at the end of the script that demonstrates how to use the `identify_region`, `check_cluster_region`, and `plot_cluster_regions` functions with a sample cluster of MNI coordinates.

```bash
python3 jhu_atlas_lookup_cluster.py
```

## Output

- **Cluster Regions**: A dictionary mapping each identified region to the number of points in that region.

```bash
> Cluster Regions: {'Anterior_corona_radiata_R': 10, 'Unclassified': 49, 'Genu_of_corpus_callosum': 1}
```

- **All Points in Same Region**: A boolean indicating whether all points in the cluster belong to the same region.

```bash
> All Points in Same Region: False
```

- **3D Visualization**: A 3D scatter plot showing the distribution of points in the cluster, color-coded by region.

<img width="727" alt="example_3d_plot" src="https://github.com/user-attachments/assets/76e7a2f4-c60d-4053-b3d6-6b3183868ea6">

## Notes

- Ensure that the voxel coordinates passed to the functions are within the bounds of the atlas data.
- Adjust the `distance_threshold` parameter in the `check_cluster_region` function to modify the sensitivity of cluster region checks.

## License

This script is open-source and available for use under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! If you have any improvements or suggestions, feel free to open an issue or submit a pull request.
