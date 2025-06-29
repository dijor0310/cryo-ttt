"""
Script to generate random subsets of spinach tomograms with monotonically increasing 
positive voxel counts.

Usage:
    python generate_spinach_subsets.py <folder_path> [--num_subsets N] [--output_file OUTPUT.yaml]
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import random


def load_nii(filepath: str) -> np.ndarray:
    """Load NIfTI file and return the data as numpy array."""
    nii_img = nib.load(filepath)
    return nii_img.get_fdata()


def count_positive_voxels(data: np.ndarray) -> int:
    """Count the number of positive voxels (value == 1) in the data."""
    return np.sum(data == 1)


def get_spinach_files(folder_path: str) -> List[Tuple[str, int]]:
    """
    Get all spinach tomogram files and their positive voxel counts.
    
    Args:
        folder_path: Path to the folder containing .nii.gz files
        
    Returns:
        List of tuples (filename, positive_voxel_count)
    """
    folder = Path(folder_path)
    spinach_files = []
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Find all .nii.gz files that start with "spinach"
    for file in folder.glob("*.nii.gz"):
        if file.name.startswith("spinach"):
            try:
                # Load the file and count positive voxels
                data = load_nii(str(file))
                positive_voxels = count_positive_voxels(data)
                
                spinach_files.append((file.name, positive_voxels))
                print(f"Processed {file.name}: {positive_voxels:,} positive voxels")
                
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
    
    if not spinach_files:
        raise ValueError(f"No spinach tomogram files found in {folder_path}")
    
    print(f"\nFound {len(spinach_files)} spinach tomogram files")
    return spinach_files


def generate_monotonic_subsets(files_with_counts: List[Tuple[str, int]], 
                              num_subsets: int = 10) -> Tuple[Dict, Dict]:
    """
    Generate random subsets with monotonically increasing positive voxel counts.
    
    Args:
        files_with_counts: List of (filename, positive_voxel_count) tuples
        num_subsets: Number of subsets to generate
        
    Returns:
        Tuple of (subsets_dict, num_pos_voxels_dict)
    """
    # Sort files by positive voxel count for reference
    sorted_files = sorted(files_with_counts, key=lambda x: x[1])
    
    # Calculate target sizes for monotonically increasing subsets
    total_positive_voxels = sum(count for _, count in files_with_counts)
    min_subset_size = total_positive_voxels // (num_subsets * 2)  # Start with smaller subsets
    max_subset_size = total_positive_voxels // 2  # Don't exceed half of total
    
    # Generate target sizes that increase monotonically
    target_sizes = np.linspace(min_subset_size, max_subset_size, num_subsets, dtype=int)
    
    subsets_dict = {}
    num_pos_voxels_dict = {}
    
    for i, target_size in enumerate(target_sizes):
        subset_name = f"spinach_subset{i + 1}"
        
        # Create a random subset that approaches the target size
        subset_files = []
        current_positive_voxels = 0
        
        # Shuffle files for randomness
        available_files = files_with_counts.copy()
        random.shuffle(available_files)
        
        for filename, positive_voxels in available_files:
            # If adding this file would exceed target by more than 20%, skip it
            if current_positive_voxels + positive_voxels > target_size * 1.2:
                continue
                
            subset_files.append(filename)
            current_positive_voxels += positive_voxels
            
            # If we're close to target size, stop adding files
            if current_positive_voxels >= target_size * 0.8:
                break
        
        # Ensure we have at least one file per subset
        if not subset_files:
            # If no files were added, add the smallest file
            smallest_file = min(files_with_counts, key=lambda x: x[1])
            subset_files = [smallest_file[0]]
            current_positive_voxels = smallest_file[1]
        
        # Add to dictionaries
        subsets_dict[subset_name] = subset_files
        num_pos_voxels_dict[subset_name] = current_positive_voxels.item()
    
    return subsets_dict, num_pos_voxels_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate random subsets of spinach tomograms with monotonically increasing positive voxel counts"
    )
    parser.add_argument("folder_path", help="Path to folder containing .nii.gz tomogram files")
    parser.add_argument("--num_subsets", type=int, default=10, 
                       help="Number of subsets to generate (default: 10)")
    parser.add_argument("--output_file", default="spinach_subsets.yaml",
                       help="Output YAML file name (default: spinach_subsets.yaml)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        print(f"Scanning folder: {args.folder_path}")
        print(f"Looking for spinach tomogram files...")
        
        # Get spinach files and their positive voxel counts
        spinach_files = get_spinach_files(args.folder_path)
        
        # Sort by positive voxel count for display
        spinach_files.sort(key=lambda x: x[1])
        
        print(f"\nSpinach files sorted by positive voxel count:")
        for filename, count in spinach_files:
            print(f"  {filename}: {count:,} positive voxels")
        
        print(f"\nGenerating {args.num_subsets} random subsets...")
        
        # Generate subsets
        subsets_dict, num_pos_voxels_dict = generate_monotonic_subsets(spinach_files, args.num_subsets)
        
        # Create the complete YAML structure
        yaml_data = {
            **subsets_dict,
            'num_pos_voxels': num_pos_voxels_dict
        }
        
        # Save to YAML
        with open(args.output_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nResults saved to: {args.output_file}")
        print(f"\nSubset summary:")
        print(f"{'Subset':<15} {'Files':<6} {'Positive Voxels':<15}")
        print("-" * 40)
        
        for subset_name in sorted(subsets_dict.keys()):
            num_files = len(subsets_dict[subset_name])
            pos_voxels = num_pos_voxels_dict[subset_name]
            print(f"{subset_name:<15} {num_files:<6} {pos_voxels:<15,}")
        
        total_pos_voxels = sum(num_pos_voxels_dict.values())
        print(f"\nTotal positive voxels across all subsets: {total_pos_voxels:,}")
        print(f"Average positive voxels per subset: {total_pos_voxels / len(num_pos_voxels_dict):,.0f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 