# Spinach Tomogram Subsets Generator

This script generates random subsets of spinach tomogram annotations with monotonically increasing positive voxel counts.

## Features

- **Spinach-only filtering**: Only processes tomogram files that start with "spinach"
- **Positive voxel counting**: Counts voxels with value == 1 in each annotation file
- **Monotonic subsets**: Generates subsets with approximately increasing positive voxel counts
- **Random selection**: Uses random selection to create diverse subsets
- **CSV output**: Saves results in a CSV file with detailed statistics
- **Reproducible**: Uses configurable random seed for reproducible results

## Usage

```bash
python generate_spinach_subsets.py <folder_path> [options]
```

### Arguments

- `folder_path`: Path to the folder containing .nii.gz tomogram annotation files

### Options

- `--num_subsets N`: Number of subsets to generate (default: 10)
- `--output_file FILE`: Output CSV file name (default: spinach_subsets.csv)
- `--seed SEED`: Random seed for reproducibility (default: 42)

### Examples

```bash
# Basic usage
python generate_spinach_subsets.py /path/to/tomograms/

# Generate 15 subsets with custom output file
python generate_spinach_subsets.py /path/to/tomograms/ --num_subsets 15 --output_file my_subsets.csv

# Use specific random seed for reproducibility
python generate_spinach_subsets.py /path/to/tomograms/ --seed 123
```

## Output

The script generates a CSV file with the following columns:

- `subset_number`: Sequential subset identifier
- `files`: Comma-separated list of filenames in the subset
- `num_files`: Number of files in the subset
- `total_positive_voxels`: Total number of positive voxels across all files in the subset
- `target_size`: Target positive voxel count for the subset
- `achievement_ratio`: Ratio of achieved vs target positive voxel count

The CSV is sorted by `total_positive_voxels` to show the monotonic progression.

## Requirements

- Python 3.6+
- nibabel
- numpy
- pandas
- pathlib (built-in)

## Algorithm

1. **File Discovery**: Scans the input folder for .nii.gz files starting with "spinach"
2. **Voxel Counting**: Loads each file and counts positive voxels (value == 1)
3. **Target Calculation**: Creates target sizes that increase monotonically
4. **Random Subset Generation**: For each target size:
   - Randomly selects files until approaching the target
   - Ensures subsets don't exceed target by more than 20%
   - Stops when reaching 80% of target or no more suitable files
5. **Output Generation**: Creates CSV with sorted results

## Error Handling

- Skips files that can't be loaded or processed
- Ensures at least one file per subset
- Provides detailed error messages for debugging
- Gracefully handles missing folders or empty results 