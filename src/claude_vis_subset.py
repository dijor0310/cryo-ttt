import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# from pyfonts import load_google_font

# font = load_google_font("CMU Serif")
import matplotlib.font_manager as fm

# List available fonts containing 'CMU'
print([f for f in fm.findSystemFonts() if "CMU" in f])

# NeurIPS/ICLR paper styling
plt.rcParams.update({
    'font.family': 'CMU Serif',
    # 'font.serif': ['Times New Roman', 'Computer Modern', 'Times'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.figsize': (3.25, 2.5),  # Single column width
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    # 'axes.grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

# Academic paper color palette (colorblind-friendly)
COLORS = {
    'before': '#1f77b4',  # Blue
    'after': '#d62728',   # Red
    'neutral': '#7f7f7f', # Gray
    'accent1': '#ff7f0e', # Orange
    'accent2': '#2ca02c', # Green
    'accent3': '#9467bd', # Purple
    'light_blue': '#aec7e8',
    'light_red': '#ff9896',
}

# Data from the table
data = {
    'S. oleacera': {
        'S. oleracea': [0.808, None], 'Synthetic': [0.26, 0.38], 'Rory': [0.62, 0.71],
        'S. pombe (VPP)': [0.14, 0.21], 'S. pombe (defocus)': [0.33, 0.35],
        'C. reinhardtii': [0.265, 0.395], 'HDCR': [0.72, 0.77]
    },
    'Synthetic': {
        'S. oleracea': [0.29, 0.32], 'Synthetic': [0.814, None], 'Rory': [0.587, 0.612],
        'S. pombe (VPP)': [0.453, 0.432], 'S. pombe (defocus)': [0.388, 0.381],
        'C. reinhardtii': [0.453, 0.497], 'HDCR': [0.682, 0.689]
    },
    'Atty': {
        'S. oleracea': [0.637, 0.625], 'Synthetic': [0.35, 0.368], 'Rory': [0.567, 0.562],
        'S. pombe (VPP)': [0.128, 0.154], 'S. pombe (defocus)': [0.291, 0.300],
        'C. reinhardtii': [0.072, 0.000], 'HDCR': [0.717, 0.721]
    },
    'Rory': {
        'S. oleracea': [0.418, 0.44], 'Synthetic': [0.226, 0.242], 'Rory': [0.88, None],
        'S. pombe (VPP)': [0.087, 0.093], 'S. pombe (defocus)': [0.217, 0.226],
        'C. reinhardtii': [0.357, 0.12], 'HDCR': [0.471, 0.462]
    }
}

def create_paper_bar_chart(data_dict, figsize=(6.5, 3.0), single_dataset=None, 
                          show_values=False, title=None):
    """
    Create a publication-ready grouped bar chart for NeurIPS/ICLR papers
    
    Parameters:
    data_dict: Dictionary of method -> dataset -> [before, after] values
    figsize: Figure size tuple (use (3.25, 2.5) for single column, (6.5, 3.0) for double)
    single_dataset: If provided, only show this dataset across all methods
    show_values: Whether to show value labels on bars
    title: Custom title for the plot
    """
    
    methods = list(data_dict.keys())
    datasets = list(next(iter(data_dict.values())).keys())
    
    if single_dataset:
        if single_dataset not in datasets:
            raise ValueError(f"Dataset {single_dataset} not found in data")
        datasets = [single_dataset]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if single_dataset:
        # Single dataset comparison across methods
        x_pos = np.arange(len(methods))
        bar_width = 0.35
        
        before_values = []
        after_values = []
        
        for method in methods:
            before_val = data_dict[method][single_dataset][0]
            after_val = data_dict[method][single_dataset][1]
            before_values.append(before_val if before_val is not None else 0)
            after_values.append(after_val if after_val is not None else 0)
        
        # Create bars with academic styling
        bars1 = ax.bar(x_pos - bar_width/2, before_values, bar_width, 
                      label='Before TTT', color=COLORS['before'], 
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        bars2 = ax.bar(x_pos + bar_width/2, after_values, bar_width,
                      label='After TTT', color=COLORS['after'],
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        
        if show_values:
            def add_value_labels(bars, values, offset=0.01):
                for bar, val in zip(bars, values):
                    if val > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                               f'{val:.2f}', ha='center', va='bottom', 
                               fontsize=7, fontweight='normal')
            
            add_value_labels(bars1, before_values)
            add_value_labels(bars2, after_values)
        
        ax.set_xlabel('Method')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        
        if title:
            ax.set_title(title)
        elif single_dataset:
            ax.set_title(f'Performance on {single_dataset}')
            
    else:
        # Multiple datasets - use different approach for paper clarity
        # Show average performance with error bars indicating range
        before_means = []
        after_means = []
        before_stds = []
        after_stds = []
        
        for method in methods:
            before_vals = [data_dict[method][ds][0] for ds in datasets 
                          if data_dict[method][ds][0] is not None]
            after_vals = [data_dict[method][ds][1] for ds in datasets 
                         if data_dict[method][ds][1] is not None]
            
            before_means.append(np.mean(before_vals) if before_vals else 0)
            after_means.append(np.mean(after_vals) if after_vals else 0)
            before_stds.append(np.std(before_vals) if len(before_vals) > 1 else 0)
            after_stds.append(np.std(after_vals) if len(after_vals) > 1 else 0)
        
        x_pos = np.arange(len(methods))
        bar_width = 0.35
        
        bars1 = ax.bar(x_pos - bar_width/2, before_means, bar_width, 
                      yerr=before_stds, label='Before TTT', 
                      color=COLORS['before'], edgecolor='black', 
                      linewidth=0.5, alpha=0.8, capsize=3, 
                      error_kw={'linewidth': 0.8})
        bars2 = ax.bar(x_pos + bar_width/2, after_means, bar_width,
                      yerr=after_stds, label='After TTT', 
                      color=COLORS['after'], edgecolor='black', 
                      linewidth=0.5, alpha=0.8, capsize=3,
                      error_kw={'linewidth': 0.8})
        
        ax.set_xlabel('Method')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Average Performance Across Datasets')
    
    ax.set_ylabel('Performance Score')
    ax.legend(frameon=True, fancybox=False, shadow=False, 
             framealpha=1.0, edgecolor='black')
    
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    return fig, ax

def create_paper_heatmap(data_dict, figsize=(6.5, 3.5), title=None):
    """
    Create a publication-ready heatmap showing performance improvements
    """
    methods = list(data_dict.keys())
    datasets = list(next(iter(data_dict.values())).keys())
    
    # Calculate improvements
    improvements = []
    for method in methods:
        method_improvements = []
        for dataset in datasets:
            before, after = data_dict[method][dataset]
            if before is not None and after is not None:
                improvement = after - before
            else:
                improvement = np.nan
            method_improvements.append(improvement)
        improvements.append(method_improvements)
    
    df = pd.DataFrame(improvements, index=methods, columns=datasets)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use academic colormap
    vmax = np.nanmax(np.abs(df.values))
    im = ax.imshow(df.values, cmap='RdBu_r', aspect='auto', 
                   vmin=-vmax, vmax=vmax)
    
    # Add colorbar with proper academic styling
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance Change', rotation=270, labelpad=12)
    cbar.outline.set_linewidth(0.5)
    
    # Set ticks and labels
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels([ds.replace(' ', '\n') if len(ds) > 10 else ds 
                        for ds in datasets], rotation=0, ha='center')
    ax.set_yticklabels(methods)
    
    # Add text annotations with appropriate contrast
    for i in range(len(methods)):
        for j in range(len(datasets)):
            value = df.iloc[i, j]
            if not np.isnan(value):
                # Choose text color based on background intensity
                text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{value:+.3f}', ha='center', va='center',
                       color=text_color, fontsize=7, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                       color='gray', fontsize=8)
    
    # Remove ticks
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(methods)))
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Test-Time Training Impact')
    
    plt.tight_layout()
    return fig, ax

def create_paper_improvement_plot(data_dict, figsize=(6.5, 2.5), title=None):
    """
    Create a scatter plot showing before vs after performance with improvement vectors
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(data_dict.keys())
    datasets = list(next(iter(data_dict.values())).keys())
    
    # Color map for different methods
    method_colors = [COLORS['before'], COLORS['after'], COLORS['accent1'], COLORS['accent2']]
    
    for i, method in enumerate(methods):
        print(method)
        before_vals = []
        after_vals = []
        valid_datasets = []
        
        for dataset in datasets:
            before, after = data_dict[method][dataset]
            if before is not None and after is not None:
                before_vals.append(before)
                after_vals.append(after)
                valid_datasets.append(dataset)
        
        if before_vals and after_vals:
            color = method_colors[i % len(method_colors)]
            
            # Plot points
            ax.scatter(before_vals, after_vals, label=method, 
                      color=color, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            
            # Draw improvement arrows (optional, comment out if too cluttered)
            # for b, a in zip(before_vals, after_vals):
            #     if abs(a - b) > 0.01:  # Only show significant changes
            #         ax.annotate('', xy=(a, a), xytext=(b, b),
            #                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.5, lw=0.8))
    
    # Add diagonal line (no improvement)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='No change')
    
    ax.set_xlabel('Dice before TTT')
    ax.set_ylabel('Dice after TTT')
    ax.legend(frameon=True, fancybox=False, shadow=False, 
             framealpha=1.0, edgecolor='black', loc='upper left')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Distribution shifts with small training sets')
    
    plt.tight_layout()
    return fig, ax

def create_paper_summary_figure(data_dict, figsize=(6.5, 4.5)):
    """
    Create a multi-panel figure suitable for paper publication
    """
    fig = plt.figure(figsize=figsize)
    
    # Create 2x2 subplot layout
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    methods = list(data_dict.keys())
    datasets = list(next(iter(data_dict.values())).keys())
    
    # Panel A: Average performance comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    before_means = []
    after_means = []
    before_sems = []
    after_sems = []
    
    for method in methods:
        before_vals = [data_dict[method][ds][0] for ds in datasets 
                      if data_dict[method][ds][0] is not None]
        after_vals = [data_dict[method][ds][1] for ds in datasets 
                     if data_dict[method][ds][1] is not None]
        
        before_means.append(np.mean(before_vals) if before_vals else 0)
        after_means.append(np.mean(after_vals) if after_vals else 0)
        before_sems.append(np.std(before_vals)/np.sqrt(len(before_vals)) if len(before_vals) > 1 else 0)
        after_sems.append(np.std(after_vals)/np.sqrt(len(after_vals)) if len(after_vals) > 1 else 0)
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x_pos - width/2, before_means, width, yerr=before_sems,
           color=COLORS['before'], alpha=0.8, capsize=2, 
           edgecolor='black', linewidth=0.5, error_kw={'linewidth': 0.8})
    ax1.bar(x_pos + width/2, after_means, width, yerr=after_sems,
           color=COLORS['after'], alpha=0.8, capsize=2,
           edgecolor='black', linewidth=0.5, error_kw={'linewidth': 0.8})
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Avg. Performance')
    ax1.set_title('(a) Average Performance', loc='left', fontweight='bold')
    
    # Panel B: Improvement distribution
    ax2 = fig.add_subplot(gs[0, 1])
    
    all_improvements = []
    for method in methods:
        for dataset in datasets:
            before, after = data_dict[method][dataset]
            if before is not None and after is not None:
                all_improvements.append(after - before)
    
    ax2.hist(all_improvements, bins=10, alpha=0.7, color=COLORS['neutral'], 
            edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color=COLORS['after'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Performance Change')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Improvement Distribution', loc='left', fontweight='bold')
    
    # Panel C: Success rate by method
    ax3 = fig.add_subplot(gs[1, 0])
    
    success_rates = []
    for method in methods:
        improvements = 0
        total = 0
        for dataset in datasets:
            before, after = data_dict[method][dataset]
            if before is not None and after is not None:
                total += 1
                if after > before:
                    improvements += 1
        success_rates.append(improvements / total if total > 0 else 0)
    
    bars = ax3.bar(range(len(methods)), success_rates, color=COLORS['accent2'], 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(0, 1)
    ax3.set_title('(c) TTT Success Rate', loc='left', fontweight='bold')
    
    # Add percentage labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=7)
    
    # Panel D: Dataset difficulty ranking
    ax4 = fig.add_subplot(gs[1, 1])
    
    dataset_avg_performance = {}
    for dataset in datasets:
        perfs = []
        for method in methods:
            before, after = data_dict[method][dataset]
            if before is not None:
                perfs.append(before)
            if after is not None:
                perfs.append(after)
        dataset_avg_performance[dataset] = np.mean(perfs) if perfs else 0
    
    sorted_datasets = sorted(dataset_avg_performance.items(), key=lambda x: x[1])
    dataset_names = [item[0] for item in sorted_datasets]
    dataset_scores = [item[1] for item in sorted_datasets]
    
    bars = ax4.barh(range(len(dataset_names)), dataset_scores, 
                    color=COLORS['accent3'], alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(dataset_names)))
    ax4.set_yticklabels([name[:10] + '...' if len(name) > 10 else name 
                        for name in dataset_names], fontsize=7)
    ax4.set_xlabel('Avg. Performance')
    ax4.set_title('(d) Dataset Difficulty', loc='left', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Example usage for paper figures
if __name__ == "__main__":
    
    # Figure 1: Main comparison (single column)
    # print("Creating main comparison figure...")
    # fig1, ax1 = create_paper_bar_chart(data, figsize=(3.25, 2.5), 
    #                                   title='Test-Time Training Performance')
    
    # # Figure 2: Detailed heatmap (double column)
    # print("Creating improvement heatmap...")
    # fig2, ax2 = create_paper_heatmap(data, figsize=(6.5, 3.0))
    
    # Figure 3: Before vs After scatter (single column)
    print("Creating improvement scatter plot...")
    fig3, ax3 = create_paper_improvement_plot(data, figsize=(3.25, 2.5))
    
    # # Figure 4: Comprehensive analysis (double column)
    # print("Creating comprehensive summary figure...")
    # fig4 = create_paper_summary_figure(data)
    
    # Show plots
    plt.show()
    
    # Save with paper-appropriate settings
    save_kwargs = {
        'dpi': 600,
        # 'bbox_inches': 'tight',
        'pad_inches': 0.05,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    
    # Uncomment to save
    # fig1.savefig('subset_charts/fig1_main_comparison.pdf', **save_kwargs)
    # fig2.savefig('subset_charts/fig2_improvement_heatmap.pdf', **save_kwargs)
    fig3.savefig('subset_charts/fig3_scatter_plot.pdf', **save_kwargs)
    # fig4.savefig('subset_charts/fig4_comprehensive_analysis.pdf', **save_kwargs)
    
    print("Figures ready for paper submission!")