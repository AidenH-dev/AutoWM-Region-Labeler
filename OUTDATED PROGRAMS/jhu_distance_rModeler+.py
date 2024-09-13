import nibabel as nib
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import pandas as pd
from scipy.spatial import distance

# Look-up table (LUT) for white matter regions, including "Unclassified"
lut = {
    0: "Unclassified",
    1: "Middle_cerebellar_peduncle",
    2: "Pontine_crossing_tract_(a_part_of_MCP)",
    3: "Genu_of_corpus_callosum",
    # (rest of the LUT remains unchanged)
}

# Pre Step 1: Prep MNI coords to read as MRIcron Coordinates
def shift_coordinates(coords, shift_values=(92, 127, 73)):
    shift = np.array(shift_values)
    shifted_coords = [tuple(np.array(coord) + shift) for coord in coords]
    return shifted_coords

# Step 1: Load and extract only the regions that contain MNI coordinates, with optional downsampling
def load_relevant_regions(file_path, relevant_region_ids, downsample_factor=2):
    nii = nib.load(file_path)
    nii_data = nii.get_fdata()

    selected_regions_data = []
    selected_colors = []

    for region_idx in relevant_region_ids:
        if region_idx == 0:  # Skip "Unclassified" region for 3D model generation
            continue
        region_data = np.zeros_like(nii_data)
        region_data[nii_data == region_idx] = 1

        if np.any(region_data):
            region_data = region_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            selected_regions_data.append(region_data)
            color = (random.random(), random.random(), random.random())
            selected_colors.append(color)

    return selected_regions_data, selected_colors, nii_data

# Step 2: Identify the white matter region corresponding to the coordinates
def identify_region(voxel_coords, atlas_data, lut):
    label_value = int(atlas_data[tuple(voxel_coords)])
    return lut.get(label_value, "Unclassified"), label_value

# Step 3: Calculate the Euclidean distance to the closest region edge
def find_nearest_region_edges(unclassified_coord, region_edges):
    distances = []
    for region_id, edge_voxels in region_edges.items():
        # Calculate the distance between the unclassified point and every voxel on the edge of the region
        dist_to_edge = np.min(np.linalg.norm(edge_voxels - unclassified_coord, axis=1))
        distances.append((region_id, lut[region_id], dist_to_edge))
    
    # Sort by distance and return the three closest regions
    distances.sort(key=lambda x: x[2])
    return distances[:3]

# Extract the surface (edge) voxels for each region
def extract_region_edges(atlas_data, relevant_region_ids):
    region_edges = {}
    for region_id in relevant_region_ids:
        if region_id == 0:  # Skip "Unclassified"
            continue
        # Create a binary mask for the region
        region_mask = atlas_data == region_id
        # Use skimage to extract the surface (edges) of the region
        edges = measure.find_contours(region_mask, 0.5)
        # Flatten and store the edge voxels for each region
        edge_voxels = np.vstack([np.array(edge) for edge in edges])
        region_edges[region_id] = edge_voxels
    return region_edges

# Step 4: Create 3D plot and model unclassified points with nearest region edges
def create_combined_3d_plot(relevant_regions_data, region_colors, mni_coords, atlas_data, relevant_region_ids, downsample_factor, region_edges):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot white matter regions
    for i, region_data in enumerate(relevant_regions_data):
        verts, faces, normals, values = measure.marching_cubes(region_data, level=0.5)
        mesh = Poly3DCollection(verts[faces], alpha=0.1, linewidths=0, edgecolors='none')
        mesh.set_facecolor(region_colors[i])
        ax.add_collection3d(mesh)

    # Plot MNI coordinates, including "Unclassified"
    for coord in mni_coords:
        region_name, region_idx = identify_region(coord, atlas_data, lut)
        downsampled_coord = [coord[i] // downsample_factor for i in range(3)]

        if region_name == "Unclassified":
            # Find the nearest three regions' edges
            nearest_regions = find_nearest_region_edges(coord, region_edges)
            print(f"Unclassified coordinate {coord} nearest regions (by edge distance):")
            for region_id, region_name, dist in nearest_regions:
                print(f"  - {region_name} (Region ID: {region_id}), Distance to edge: {dist:.2f}")

        ax.scatter(*downsampled_coord, color='red' if region_name == "Unclassified" else 'blue', s=50)
        ax.text(downsampled_coord[0], downsampled_coord[1], downsampled_coord[2], f'{region_name}', fontsize=8)

    ax.set_xlim(0, atlas_data.shape[0] // downsample_factor)
    ax.set_ylim(0, atlas_data.shape[1] // downsample_factor)
    ax.set_zlim(0, atlas_data.shape[2] // downsample_factor)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=45, azim=45)

    plt.show()

# Main function
def main():
    file_path = '../AutoWM-Region-Labeler/JHU_atlas/JHU-WhiteMatter-labels-1mm.nii.gz'  # Update with your local path
    downsample_factor = 1  # Downsampling factor

    mni_coords = [(12, -91, 11), (27, -88, 20), (-9, -28, -2)]  # Example MNI coordinates

    shifted_coordinates = shift_coordinates(mni_coords)

    atlas_img = nib.load(file_path)
    atlas_data = atlas_img.get_fdata()

    relevant_region_ids = set()
    for coord in shifted_coordinates:
        _, region_id = identify_region(coord, atlas_data, lut)
        relevant_region_ids.add(region_id)

    relevant_regions_data, region_colors, _ = load_relevant_regions(file_path, relevant_region_ids, downsample_factor)

    # Extract edges for all relevant regions
    region_edges = extract_region_edges(atlas_data, relevant_region_ids)

    create_combined_3d_plot(relevant_regions_data, region_colors, shifted_coordinates, atlas_data, relevant_region_ids, downsample_factor, region_edges)

if __name__ == "__main__":
    main()
