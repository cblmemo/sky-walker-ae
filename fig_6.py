from plot_utils import *

def fig6():
    InitMatplotlib(10, 7)
    name2group = {
        'Cross-User\nSharing': {
            0.43613963157590807: 'CH',
            0.5222082837640203: 'Optimal',
        },
        'Bursty\nRequest': {
            0.839560427106233: 'CH',
            0.90335743590906: 'Optimal',
        },
        'Heterogeneous\nProgram': {
            0.8240817940136786: 'CH',
            0.90335743590906: 'Optimal',
        },
    }
    palette = sns.color_palette('colorblind', 3)
    
    # Collect data
    groups = list(name2group.keys())
    labels = ['CH', 'Optimal']
    labels_mapping = {
        'CH': 'Consistent Hashing',
        'Optimal': 'Optimal',
    }
    
    # Prepare data for grouped bar chart
    cache_hit_rates = []
    legend_elements = []
    new_name2group = {}
    
    for group_name in groups:
        group_data = name2group[group_name]
        group_hit_rates = []
        populate_legend_elements = not legend_elements
        for i, label in enumerate(labels):
            for hit_rate, label_in_exp in group_data.items():
                if label_in_exp != label:
                    continue
                group_hit_rates.append(hit_rate)
                if populate_legend_elements:
                    legend_elements.append(
                        Patch(facecolor='White', 
                            edgecolor=palette[i], 
                            label=labels_mapping[label],
                            linewidth=0.8))
        cache_hit_rates.append(group_hit_rates)
    
    # Set up plot parameters
    fig_width, fig_height = plt.gcf().get_size_inches()
    x = np.arange(len(groups))
    width = 0.25  # Width of the bars
    print(new_name2group)
    
    # Create hit rate grouped bar plot
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    for i, label in enumerate(labels):
        values = [group[i] for group in cache_hit_rates]
        plt.bar(x + (i - 1) * width, values, width, label=label, color=palette[i])
    
    plt.ylabel('Cache Hit Rate (%)')
    plt.xticks(x, groups)
    plt.yticks(
        np.arange(0, plt.ylim()[1], 0.2),
        [f'{int(val*100)}' for val in np.arange(0, plt.ylim()[1], 0.2)]
    )
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1), 
               ncol=3, fontsize=9, handletextpad=0.5, columnspacing=1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    print(plt.ylim())
    plt.ylim(bottom=0.20)
    plt.tight_layout()
    plt.savefig(f'{paper_fig_dir}/fig-6.pdf', dpi=300)

if __name__ == '__main__':
    fig6()
