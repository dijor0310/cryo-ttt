import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Set the path to your dataset
DATA_PATH = "/media/ssd3/diyor/membrain-seg-data/MemBrain_seg_training_data/labelsTr"

def get_file_type(filename):
    """Extract and process the type from filename"""
    base_type = filename.split('_')[0]
    
    # Discard 'virly' type
    if base_type in []:
        return None
    
    # Merge 'cts' and 'polnet' into 'synthetic'
    if base_type in ['cts', 'polnet']:
        return 'synthetic'
    
    return base_type

def analyze_files():
    """Analyze all files and return a DataFrame with their statistics"""
    stats = []
    
    for file in Path(DATA_PATH).glob('*.nii.gz'):
        file_type = get_file_type(file.name)
        if file_type is None:  # Skip 'virly' type
            continue
            
        # Load the NIfTI file
        nifti_img = nib.load(str(file))
        data = nifti_img.get_fdata()
        
        # Calculate positive voxels
        positive_voxels = np.sum(data == 1)
        
        stats.append({
            'file_type': file_type,
            'filename': file.name,
            'positive_voxels': positive_voxels
        })
    
    return pd.DataFrame(stats)

def get_type_statistics(df):
    """Calculate total positive voxels per type"""
    type_stats = df.groupby('file_type').agg({
        'positive_voxels': 'sum',
        'filename': 'count'
    }).rename(columns={'filename': 'file_count'})
    return type_stats

def create_balanced_subsets(df):
    """Create balanced subsets using smallest type's total as target size"""
    # Calculate total positive voxels per type
    type_stats = get_type_statistics(df)
    
    # Find the smallest type's total positive voxels
    target_size = type_stats['positive_voxels'].min()
    target_size = 4_000_000
    smallest_type = type_stats['positive_voxels'].idxmin()
    
    print(f"Using {smallest_type} as reference type")
    print(f"Target subset size: {target_size:,.0f} positive voxels")
    
    subsets = []
    subset_number = 1
    
    # Process each type separately
    for file_type in df['file_type'].unique():
        # type_files = df[df['file_type'] == file_type].sort_values('positive_voxels', ascending=False)
        type_files = df[df['file_type'] == file_type]
        
        current_subset = []
        current_positive_voxels = 0
        
        for _, row in type_files.iterrows():
            # If adding this file would exceed target size by more than 10%, start a new subset
            if current_positive_voxels + row['positive_voxels'] > target_size * 1.1 and current_subset:
                subsets.append({
                    'subset_number': subset_number,
                    'file_type': file_type,
                    'files': ','.join(current_subset),
                    'total_positive_voxels': current_positive_voxels,
                    'target_size': target_size
                })
                subset_number += 1
                current_subset = []
                current_positive_voxels = 0
            
            current_subset.append(row['filename'])
            current_positive_voxels += row['positive_voxels']
        
        # Add the last subset if it's not empty
        if current_subset:
            subsets.append({
                'subset_number': subset_number,
                'file_type': file_type,
                'files': ','.join(current_subset),
                'total_positive_voxels': current_positive_voxels,
                'target_size': target_size
            })
            subset_number += 1
    
    return pd.DataFrame(subsets)

def main():
    print("Analyzing files...")
    df = analyze_files()
    
    # Print initial type statistics
    type_stats = get_type_statistics(df)
    print("\nInitial statistics per type:")
    print(type_stats)
    
    print("\nCreating balanced subsets...")
    subsets_df = create_balanced_subsets(df)
    
    # Save to CSV
    output_file = 'balanced_subsets.csv'
    subsets_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nSummary of created subsets:")
    print(f"Total number of subsets: {len(subsets_df)}")
    
    print("\nSubsets per type:")
    type_counts = subsets_df.groupby('file_type').agg({
        'subset_number': 'count',
        'total_positive_voxels': ['mean', 'std', 'min', 'max']
    }).round(2)
    print(type_counts)
    
    print("\nDetailed statistics per type:")
    for file_type in subsets_df['file_type'].unique():
        type_subsets = subsets_df[subsets_df['file_type'] == file_type]
        print(f"\n{file_type}:")
        print(f"Number of subsets: {len(type_subsets)}")
        print(f"Average positive voxels per subset: {type_subsets['total_positive_voxels'].mean():.2f}")
        print(f"Standard deviation: {type_subsets['total_positive_voxels'].std():.2f}")
        print(f"Min positive voxels: {type_subsets['total_positive_voxels'].min():.2f}")
        print(f"Max positive voxels: {type_subsets['total_positive_voxels'].max():.2f}")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()