import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# Set the path to your dataset
DATA_PATH = "/media/ssd3/diyor/membrain-seg-data/MemBrain_seg_training_data/labelsTr"

def get_file_type(filename):
    """Extract the type from filename (substring before first '_')"""
    return filename.split('_')[0]

def analyze_nifti_files():
    # Initialize storage for statistics
    stats = defaultdict(list)
    
    # Process each .nii.gz file
    for file in Path(DATA_PATH).glob('*.nii.gz'):
        file_type = get_file_type(file.name)
        
        # Load the NIfTI file
        nifti_img = nib.load(str(file))
        data = nifti_img.get_fdata()
        
        # Calculate statistics
        positive_voxels = np.sum(data == 1)
        total_voxels = data.size
        positive_ratio = positive_voxels / total_voxels
        
        # Store statistics
        stats['file_type'].append(file_type)
        stats['filename'].append(file.name)
        stats['positive_voxels'].append(positive_voxels)
        stats['total_voxels'].append(total_voxels)
        stats['positive_ratio'].append(positive_ratio)
    
    # Convert to DataFrame
    df = pd.DataFrame(stats)
    
    # Calculate summary statistics
    summary = df.groupby('file_type').agg({
        'filename': 'count',
        'positive_voxels': ['sum', 'mean', 'std'],
        'positive_ratio': ['mean', 'std']
    }).round(2)
    
    # Rename columns for clarity
    summary.columns = ['file_count', 'total_positive_voxels', 'mean_positive_voxels', 
                      'std_positive_voxels', 'mean_positive_ratio', 'std_positive_ratio']
    
    return df, summary

def create_visualizations(df, summary):
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. File count per type
    plt.subplot(2, 2, 1)
    sns.barplot(x=summary.index, y='file_count', data=summary)
    plt.title('Number of Files per Type')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 2. Total positive voxels per type
    plt.subplot(2, 2, 2)
    sns.barplot(x=summary.index, y='total_positive_voxels', data=summary)
    plt.title('Total Positive Voxels per Type')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 3. Box plot of positive voxels distribution
    plt.subplot(2, 2, 3)
    sns.boxplot(x='file_type', y='positive_voxels', data=df)
    plt.title('Distribution of Positive Voxels per Type')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 4. Box plot of positive ratio distribution
    plt.subplot(2, 2, 4)
    sns.boxplot(x='file_type', y='positive_ratio', data=df)
    plt.title('Distribution of Positive Voxel Ratio per Type')
    plt.xticks(rotation=45)
    plt.ylabel('Ratio')
    
    plt.tight_layout()
    plt.savefig('nifti_analysis_plots.png', dpi=300, bbox_inches='tight')
    
    # Save summary statistics to CSV
    summary.to_csv('nifti_analysis_summary.csv')

def main():
    print("Analyzing NIfTI files...")
    df, summary = analyze_nifti_files()
    
    print("\nSummary Statistics:")
    print(summary)
    
    print("\nCreating visualizations...")
    create_visualizations(df, summary)
    print("Analysis complete! Results saved to 'nifti_analysis_plots.png' and 'nifti_analysis_summary.csv'")

if __name__ == "__main__":
    main()