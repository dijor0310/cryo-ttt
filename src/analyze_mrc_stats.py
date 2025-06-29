import os
import numpy as np
import mrcfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the path to your dataset
DATA_PATH = "/mnt/hdd_pool_zion/userdata/diyor/data/deepict/DEF/labels"

def analyze_mrc_files():
    # Initialize storage for statistics
    stats = []
    
    # Process each .mrc file
    for file in Path(DATA_PATH).glob('*.mrc'):
        # Load the MRC file
        if not file.name.endswith("membranes.mrc"):
            print(file.name)
            continue
        with mrcfile.open(str(file)) as mrc:
            data = mrc.data
        
        # Calculate statistics
        positive_voxels = np.sum(data == 1)
        print(positive_voxels)
        total_voxels = data.size
        positive_ratio = positive_voxels / total_voxels
        
        # Store statistics
        stats.append({
            'filename': file.name,
            'positive_voxels': positive_voxels,
            'total_voxels': total_voxels,
            'positive_ratio': positive_ratio
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(stats)
    
    # Calculate summary statistics
    summary = {
        'total_files': len(df),
        'total_positive_voxels': df['positive_voxels'].sum(),
        'mean_positive_voxels': df['positive_voxels'].mean(),
        'std_positive_voxels': df['positive_voxels'].std(),
        'min_positive_voxels': df['positive_voxels'].min(),
        'max_positive_voxels': df['positive_voxels'].max(),
        'mean_positive_ratio': df['positive_ratio'].mean(),
        'std_positive_ratio': df['positive_ratio'].std()
    }
    
    return df, pd.Series(summary).round(2)

def create_visualizations(df, summary):
    # Set style
    plt.style.use('seaborn')
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Histogram of positive voxels
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='positive_voxels', bins=30)
    plt.title('Distribution of Positive Voxels')
    plt.xlabel('Number of Positive Voxels')
    plt.ylabel('Count')
    
    # 2. Histogram of positive ratio
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='positive_ratio', bins=30)
    plt.title('Distribution of Positive Voxel Ratio')
    plt.xlabel('Ratio of Positive Voxels')
    plt.ylabel('Count')
    
    # 3. Box plot of positive voxels
    plt.subplot(2, 2, 3)
    sns.boxplot(y=df['positive_voxels'])
    plt.title('Box Plot of Positive Voxels')
    plt.ylabel('Number of Positive Voxels')
    
    # 4. Box plot of positive ratio
    plt.subplot(2, 2, 4)
    sns.boxplot(y=df['positive_ratio'])
    plt.title('Box Plot of Positive Voxel Ratio')
    plt.ylabel('Ratio of Positive Voxels')
    
    plt.tight_layout()
    plt.savefig('mrc_analysis_plots.png', dpi=300, bbox_inches='tight')
    
    # Save summary statistics to CSV
    summary.to_csv('mrc_analysis_summary.csv', header=['value'])

def main():
    print("Analyzing MRC files...")
    df, summary = analyze_mrc_files()
    
    print("\nSummary Statistics:")
    print(summary)
    
    print("\nCreating visualizations...")
    create_visualizations(df, summary)
    print("Analysis complete! Results saved to 'mrc_analysis_plots.png' and 'mrc_analysis_summary.csv'")

if __name__ == "__main__":
    main()