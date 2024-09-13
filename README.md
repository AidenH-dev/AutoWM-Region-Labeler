<h1 align="center">
  Auto White Matter Region Labeler
</h1>

<p align="center">
Automate the region labeling for characterizing and comparing brain activations in white matter and gray matter during different cognitive states.
</p>

## Overview

This project provides a set of tools for identifying, visualizing, and analyzing white matter regions in the brain using the JHU White Matter Atlas. The tools range from simple region identification to more complex visualization of white matter tracts and 3D modeling. The project includes three main scripts:

1. **`jhu_3d+.py`**: Generates 3D visualizations of white matter regions with MNI coordinates and vector plots.
2. **`jhu_R+DV.py`**: Identifies the closest white matter regions to specific coordinates and calculates distances and vectors between them.
3. **`jhu_tract_modeler.py`**: Creates an interactive 3D model of white matter regions using a web interface powered by Plotly and Dash.

## Prerequisites

Ensure you have the following Python packages installed:

- `nibabel`: For loading neuroimaging data in NIfTI format.
- `numpy`: For numerical operations.
- `scipy`: For spatial distance calculations.
- `matplotlib` and `plotly`: For generating visualizations.
- `skimage`: For creating 3D meshes from volumetric data.
- `dash`: For building an interactive web application for visualizing white matter tracts.

You can install these packages using pip:

```bash
pip install nibabel numpy scipy matplotlib plotly scikit-image dash
```

## Files

- `JHU-WhiteMatter-labels-1mm.nii.gz`: The NIfTI file containing the JHU White Matter Atlas.
- `JHU-WhiteMatter-labels-2mm.nii.gz`: A downsampled version of the JHU White Matter Atlas.
- `jhu_3d+.py`: Script for advanced 3D visualization.
- `jhu_R+DV.py`: Script for region and distance/vector calculation.
- `jhu_tract_modeler.py`: Script for interactive 3D tract modeling.

## Scripts Overview

### 1. `jhu_3d+.py`

This script provides advanced 3D visualization of white matter regions in the brain. It integrates MNI coordinate processing, vector plotting, and region identification.

**Key Features:**
- Loads the JHU White Matter Atlas and extracts regions.
- Converts MNI coordinates to MRIcron space.
- Identifies the white matter regions closest to given voxel coordinates.
- Generates a 3D plot with MNI coordinates and vector plots showing proximity to nearby regions.

### 2. `jhu_R+DV.py`

This script processes MNI coordinates, shifts them into MRIcron space, and identifies the closest white matter regions for each point. It calculates the Euclidean distances and vectors to the three nearest regions for further analysis.

**Key Features:**
- Shifts MNI coordinates for atlas alignment.
- Identifies the closest regions for each coordinate.
- Outputs results to a CSV file with regions, distances, and vectors.
- Includes a formatted CSV output with each voxel's 3 closest regions.

### 3. `jhu_tract_modeler.py`

This script generates a fully interactive 3D model of white matter tracts using Plotly and Dash. It allows the user to interact with the model through a web-based interface and select specific white matter regions for visualization.

**Key Features:**
- Loads the JHU White Matter Atlas and extracts relevant regions.
- Generates 3D meshes using marching cubes to create tract models.
- Creates an interactive Plotly-based 3D visualization.
- Includes a Dash web application with a region selector for exploring specific tracts in the model.

## Usage

### 1. `jhu_3d+.py`

This script loads the JHU White Matter Atlas, processes MNI coordinates, and generates a 3D plot. Example usage:

```bash
python jhu_3d+.py
```

### 2. `jhu_R+DV.py`

This script processes MNI coordinates and outputs the results in CSV format. Example usage:

```bash
python jhu_R+DV.py
```

### 3. `jhu_tract_modeler.py`

This script launches a Dash web application for visualizing white matter tracts interactively. Example usage:

```bash
python jhu_tract_modeler.py
```

After running, visit the local server in your browser to explore the interactive 3D model.

## Output

- **CSV File**: For `jhu_R+DV.py`, the results are written to a CSV file, which includes the input coordinates, identified regions, and vectors.
- **3D Visualization**: For `jhu_3d+.py` and `jhu_tract_modeler.py`, the scripts generate interactive 3D models and visualizations.
  
## License

This project is open-source and available for use under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with any improvements or suggestions.
---
