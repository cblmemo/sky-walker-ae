import matplotlib.pyplot as plt
import numpy as np
from plot_utils import *

def plot_cost_comparison():
    # Initialize matplotlib
    InitMatplotlib(11.5,7)
    
    # Data
    costs = [236757.5, 179112, 106632]
    labels = ['On-Demand\nAutoscaling', 'Region\nLocal', 'Aggregated']
    user2cost = {'us-east-1': 56184, 'us-west': 49152, 'eu-west': 46224, 'eu-central': 15096, 'us-east-2': 12456}
    
    # Calculate reduction ratio
    reduction_ratio = (costs[1] - costs[2]) / costs[1] * 100
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width * 0.9, fig_height * 1.6), dpi=300)
    
    # Plot stacked bar for Local costs
    bar_width = 0.5
    bar_positions = [0, 1, 2]
    
    # Get user costs and sort them for stacking
    users = list(user2cost.keys())
    user_costs = [user2cost[user] for user in users]
    
    # Create color palette
    palette = sns.color_palette('colorblind', len(users))

    ax.bar(bar_positions[0], costs[0], width=bar_width, color='#EF5350')
    
    # Plot stacked bar for Local
    bottom = 0
    for i, (user, cost) in enumerate(zip(users, user_costs)):
        ax.bar(bar_positions[1], cost, width=bar_width, bottom=bottom, color=palette[i], label=user)
        bottom += cost
    
    # Plot single bar for Aggregated
    ax.bar(bar_positions[2], costs[2], width=bar_width, color='#4CAF50')
    
    # Add labels
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Estimated Cost (\$)')
    
    # Format y-axis with scientific notation
    y_scale = 1000
    cost_max = max(costs)
    scaled_cost_max = int(cost_max / y_scale) + 1
    cost_ticks = list(range(0, scaled_cost_max + 1, 50))
    ax.set_yticks([y * y_scale for y in cost_ticks])
    ax.set_yticklabels(cost_ticks)
    ax.set_yticks([])
    # ax.text(ax.get_xlim()[0] - 0.2, cost_max*1.05, '$\\times 10^3$', va='bottom', ha='center')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add arrow and reduction text
    arrow_start = (1, costs[1]*1.05)
    arrow_end = (2, costs[2])
    ax.annotate('', 
                xy=arrow_end, 
                xytext=arrow_start,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.8),
                ha='center', va='bottom', fontsize=10)
    ax.text(2, costs[1] * 0.82, f'{reduction_ratio:.1f}%\nreduction', 
            ha='center', va='bottom')

    ax.text(0, costs[0] * 1.03, f'{costs[0] / costs[2]:.1f}x of\n{labels[2]}',
            ha='center', va='bottom')

    # ax.set_xlabel('Systems')
    
    # Add legend
    # ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{paper_fig_dir}/fig-3-b.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_cost_comparison()
