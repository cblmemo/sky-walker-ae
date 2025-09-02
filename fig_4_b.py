from plot_utils import *

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(file_path):
    """Load data from the JSONL file."""
    times = []
    replica1_usage = []
    replica2_usage = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            times.append(data['t'])
            usages = data['usage']
            urls = list(usages.keys())
            replica1_usage.append(usages[urls[0]] * 100)
            replica2_usage.append(usages[urls[1]] * 100)
    
    # Convert to numpy arrays for easier manipulation
    times = np.array(times)
    # Normalize times to start from 0
    times = times - times[0]
    replica1_usage = np.array(replica1_usage)
    replica2_usage = np.array(replica2_usage)
    
    time_filter = (times >= 100) & (times <= 180)
    times = times[time_filter] - times[time_filter][0]
    replica1_usage = replica1_usage[time_filter]
    replica2_usage = replica2_usage[time_filter]
    
    return times, replica1_usage, replica2_usage, urls

def plot_load_imbalance(file_path='usage.jsonl'):
    """Create a figure showing the load imbalance ratio."""
    times, replica1_usage, replica2_usage, urls = load_data(file_path)
    
    # Calculate imbalance ratio
    max_usage = np.maximum(replica1_usage, replica2_usage)
    min_usage = np.minimum(replica1_usage, replica2_usage)
    # Avoid division by zero
    ratio = np.zeros_like(max_usage)
    nonzero_indices = min_usage > 0
    ratio[nonzero_indices] = replica1_usage[nonzero_indices] / replica2_usage[nonzero_indices]
    
    # Print the maximum ratio difference
    max_ratio = np.max(ratio)
    print(f'Maximum ratio difference (R1/R2): {max_ratio:.2f}')
    
    # Create figure for imbalance ratio
    plt.figure(figsize=(fig_width, fig_height), dpi=300)

    palette = sns.color_palette('colorblind', 3)
    # Plot imbalance ratio
    plt.plot(times, ratio, '-', color=palette[0])
    plt.ylabel('Memory Usage\nComparison (R1/R2)')
    plt.xlabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Create a second figure for individual replica usage
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    # Create legend elements with patches
    legend_elements = []
    for i, label in enumerate(['Replica 1', 'Replica 2']):
        legend_elements.append(
            Patch(facecolor='White', 
                edgecolor=palette[i], 
                label=label,
                linewidth=0.8))
    
    plt.plot(times, replica1_usage, '-', color=palette[0])
    plt.plot(times, replica2_usage, '-', color=palette[1])
    plt.ylabel('Memory\nUtilization (%)')
    plt.xlabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(handles=legend_elements, loc='upper left', frameon=True, 
               framealpha=0.9, edgecolor='lightgray', bbox_to_anchor=(-0.03, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{paper_fig_dir}/fig-4-b.pdf')

if __name__ == '__main__':
    InitMatplotlib(11, 7)
    plot_load_imbalance()
