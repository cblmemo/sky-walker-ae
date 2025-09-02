import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from plot_utils import *

InitMatplotlib(13,7)

# Data preparation
data = {
    'ChatBot\nArena': {
        'within': 0.2050,
        'cross': 0.0831,
    },
    'WildChat\nUser': {
        'within': 0.1901,
        'cross': 0.0255,
    },
    'WildChat\nRegion': {
        'within-region': 0.1093,
        'cross-region': 0.0254,
    },
}

# Extract data for plotting and convert to percentage
datasets = list(data.keys())
within_values = [data[d].get('within', data[d].get('within-region', 0)) * 100 for d in datasets]
cross_values = [data[d].get('cross', data[d].get('cross-region', 0)) * 100 for d in datasets]

# Set up the figure
fig, ax = plt.subplots(figsize=(fig_width * 1.5, fig_width))

# Set width of bars
bar_width = 0.4
x = np.arange(len(datasets))

palette = sns.color_palette("colorblind", 4)

# Create bars
within_bars = ax.bar(x[0:2] - bar_width/2, within_values[0:2], bar_width, label='Within', color=palette[0])
cross_bars = ax.bar(x[0:2] + bar_width/2, cross_values[0:2], bar_width, label='Cross', color=palette[1])

# Plot the third dataset with different colors
within_region_bar = ax.bar(x[2] - bar_width/2, within_values[2], bar_width, label='Within-Region', color=palette[2])
cross_region_bar = ax.bar(x[2] + bar_width/2, cross_values[2], bar_width, label='Cross-Region', color=palette[3])

# Add value labels on top of bars - ensure they're centered on the bars
for i, v in enumerate(within_values):
    bar_center = x[i] - bar_width/2
    ax.text(bar_center, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
for i, v in enumerate(cross_values):
    bar_center = (x[i] + bar_width/2) * 1.
    ax.text(bar_center, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

# Add labels and title
ax.set_ylabel('Similarity (%)')
ax.set_xticks(x)
ax.set_xticklabels(datasets)

# Create legend elements
legend_elements = [
    Patch(facecolor=palette[0], label='Within-User'),
    Patch(facecolor=palette[1], label='Across-User'),
    Patch(facecolor=palette[2], label='Within-Region'),
    Patch(facecolor=palette[3], label='Across-Region'),
]
# Add legend above the plot
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.05), 
          frameon=True, edgecolor='lightgray', framealpha=0.9, borderpad=0.3, ncol=2, fontsize=13)

plt.tight_layout()
plt.savefig(f'{paper_fig_dir}/fig-5-a.pdf', bbox_inches='tight')
plt.close()
