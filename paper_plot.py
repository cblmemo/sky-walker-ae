from plot_utils import *
from pathlib import Path
import json

paper_fig_dir = 'paper_fig'
os.makedirs(paper_fig_dir, exist_ok=True)

result_file_dir = Path(__file__).parent
TIMEOUT_SKIP_SECONDS = 1000

SYS_NAME = 'Walker'
SKY_WALKER_CH_NAME = f'Sky\n{SYS_NAME}\nCH'
SKY_WALKER_NAME = f'Sky\n{SYS_NAME}'

desc2name = {
    'gke_gateway': 'GKE\nGateway',
    'sky_round_robin': 'RR',
    'sky_least_load': 'LL',
    'sky_consistent_hashing': 'CH',
    'sgl': 'SGL',
    'sky_walker_ch': SKY_WALKER_CH_NAME,
    'sky_walker_prefix': SKY_WALKER_NAME,
}

def get_metric_for_group(group_name):
    metrics_path = Path(f'{result_file_dir}/metric/{group_name}')
    tot_reqs = 0
    tot_failures = 0
    metrics = []
    if metrics_path.is_dir():
        # If it's a directory, read all json files inside
        for json_file in metrics_path.glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                file_metrics = [m for m in data['metrics'] if m['e2e_latency'] is not None and m['e2e_latency'] < TIMEOUT_SKIP_SECONDS and m['failed'] is None]
                tot_reqs += len(data['metrics'])
                tot_failures += len(data['metrics']) - len(file_metrics)
                metrics.extend(file_metrics)
    else:
        # If it's a file, read it directly
        metrics_file = Path(f'{result_file_dir}/metric/{group_name}.json')
        if not metrics_file.exists():
            raise FileNotFoundError(f'{metrics_file} or {metrics_path} does not exist')
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            file_metrics = [m for m in data['metrics'] if m['e2e_latency'] is not None and m['e2e_latency'] < TIMEOUT_SKIP_SECONDS and m['failed'] is None]
            metrics = file_metrics
            tot_reqs += len(data['metrics'])
            tot_failures += len(data['metrics']) - len(file_metrics)
    return metrics, tot_failures / tot_reqs

def _get_tpt(metrics):
    st = min(m['start'] for m in metrics)
    et = max(m['end'] for m in metrics)
    num_tokens = sum(m['input_tokens'] + 2 * m['output_tokens'] 
            for m in metrics if m['input_tokens'] is not None and m['output_tokens'] is not None)
    return num_tokens / (et - st)

def _get_ttft(metrics):
    return [m['ttft'] for m in metrics if m['ttft'] is not None]

def _get_e2e_latency(metrics):
    return [m['e2e_latency'] for m in metrics if m['e2e_latency'] is not None]

def _get_cache_hit_rate(metrics):
    tot_hit = 0
    tot_input = 0
    for m in metrics:
        if m['input_tokens'] is not None and m['cached_tokens'] is not None:
            tot_input += m['input_tokens']
            tot_hit += min(m['cached_tokens'], m['input_tokens'])
    return tot_hit / tot_input

def eval_plot_one(idx, gn2alias, fig_width_ratio=1, exp_name_swap_dict=None, scale_tpt=False):

    def _get_figure_name(suffix):
        suffix2idx = {
            'tpt': 0,
            'ttft': 1,
            'e2e_latency': 2,
        }
        if idx < 0:
            idx_in_fig_9 = suffix2idx[suffix]
            chara = chr(ord('a') + idx_in_fig_9)
            return f'fig-9-{chara}'
        idx_in_fig_8 = suffix2idx[suffix] * 4 + idx
        chara = chr(ord('a') + idx_in_fig_8)
        return f'fig-8-{chara}'
    
    # Collect data for all groups
    tpts = []
    ttfts = []
    e2e_latencies = []
    cache_hit_rates = []
    labels = []
    
    for raw_gn, name in gn2alias.items():
        gns = [raw_gn]
        if exp_name_swap_dict is not None:
            for k, vv in exp_name_swap_dict.items():
                if isinstance(vv, str):
                    vl = [vv]
                else:
                    vl = vv
                for v in vl:
                    gns.append(raw_gn.replace(k, v))
        metrics = []
        for gn in gns:
            one_metrics = get_metric_for_group(gn)[0]
            metrics.extend(one_metrics)
        tpt = _get_tpt(metrics)
        if scale_tpt:
            tpt /= (12 if 'no_cross_region' in raw_gn else 9)
        ttft = _get_ttft(metrics)
        e2e_latency = _get_e2e_latency(metrics)
        tpts.append(tpt)
        ttfts.append(ttft)
        e2e_latencies.append(e2e_latency)
        labels.append(name)
        cache_hit_rates.append(_get_cache_hit_rate(metrics))
    
    # Create TPT bar plot with broken y-axis
    fig = plt.figure(figsize=(fig_width*fig_width_ratio, fig_height), dpi=300)
    
    # Control the ratio between the two subplots
    bottom_ratio = 0.05  # Proportion of figure height for bottom plot
    top_ratio = 0.6      # Proportion of figure height for top plot
    gap_ratio = 0.2 - bottom_ratio  # Remaining space between plots and for margins
    
    # Adjust the ratio between the two subplots (make top plot larger)
    ax1 = fig.add_axes([0.1, bottom_ratio + gap_ratio, 0.8, top_ratio])  # top plot
    ax2 = fig.add_axes([0.1, 0.1, 0.8, bottom_ratio])  # bottom plot
    
    palette = sns.color_palette('colorblind', len(tpts))
    
    # Plot on both axes
    ax1.bar(range(len(tpts)), tpts, color=palette)
    ax2.bar(range(len(tpts)), tpts, color=palette)
    
    # Set the y-axis limits for each subplot
    y_min = min(tpts) * 0.95
    y_max = max(tpts) * 1.05

    ax2_ymax = y_min * 0.2
    
    # Top plot shows the upper part
    ax1.set_ylim(bottom=y_min, top=y_max)
    # Bottom plot shows a small portion near zero
    ax2.set_ylim(0, ax2_ymax)
    
    # Hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False, bottom=False)
    
    # Adjust only the vertical component of the diagonal lines
    d = .008  # horizontal component (same for both)
    d_top_vert = d  # vertical component for top plot
    # Scale the bottom diagonal lines based on the ratio of plot heights
    d_bottom_vert = d_top_vert * (top_ratio / bottom_ratio)  # adjust vertical component for bottom plot
    
    # Draw the break marks on the bottom of the top plot
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax1.plot((-d, +d), (-d_top_vert, +d_top_vert), **kwargs)
    ax1.plot((1-d, 1+d), (-d_top_vert, +d_top_vert), **kwargs)
    
    # Draw the break marks on the top of the bottom plot
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d_bottom_vert, 1+d_bottom_vert), **kwargs)
    ax2.plot((1-d, 1+d), (1-d_bottom_vert, 1+d_bottom_vert), **kwargs)
    
    # Labels and ticks
    ax2.set_xticks(range(len(tpts)))
    ax2.set_xticklabels(labels)
    ratio = 0.07 if abs(1-fig_width_ratio) > 0.01 else 0.05
    fig.text((1.-1./fig_width_ratio) * ratio, 0.4, 'Throughput (token/s)', va='center', rotation='vertical')
    
    # Scale for y-axis (divide by 1000)
    y_scale = 1000
    
    # Top plot y-ticks
    scaled_ymax = int((y_max) / y_scale)
    step = 1 if y_max < 10000 else 3
    scaled_y_min = int(y_min / y_scale) // step * step  # Make scaled_y_min a multiple of step
    yticks_top = list(range(scaled_y_min, scaled_ymax + 1, step))
    # if yticks_top[-1] * y_scale > max(tpts):
    #     yticks_top.pop()
    ax1.set_yticks([y * y_scale for y in yticks_top])
    ax1.set_yticklabels(yticks_top)
    
    # Bottom plot y-ticks
    # Bottom plot y-ticks with the same step as top plot
    scaled_y_min_bottom = int(ax2_ymax / y_scale)
    yticks_bottom = list(range(0, scaled_y_min_bottom + 1, step))
    ax2.set_yticks([y * y_scale for y in yticks_bottom])
    ax2.set_yticklabels(yticks_bottom)
    
    # Add the x10^3 label to the top of y-axis
    ax1.text(-0.1, yticks_top[-1] * y_scale * 1.05, '$\\times 10^3$', va='bottom', ha='left', transform=ax1.get_yaxis_transform())
    
    # Grid lines
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Use bbox_inches='tight' instead of tight_layout to avoid the warning
    plt.savefig(f'{paper_fig_dir}/{_get_figure_name("tpt")}.pdf', bbox_inches='tight')
    plt.close()

    def _plot_latency_box(latencies, label, suffix, do_log_scale=True):
        # Create TTFT box plot
        plt.figure(figsize=(fig_width*fig_width_ratio, fig_height), dpi=300)
        # Create a DataFrame for seaborn
        data = []
        for i, latency_list in enumerate(latencies):
            for latency_val in latency_list:
                data.append({'Group': labels[i], 'Latency': latency_val})
        df = pd.DataFrame(data)
        
        # Plot with seaborn
        # Create a custom boxplot for each group with appropriate edge colors
        ax = plt.gca()
        single_lb_max = 0
        sw_min = float('inf')
        gke_max = 0
        for i, group in enumerate(df['Group'].unique()):
            group_data = df[df['Group'] == group]
            p50 = np.percentile(group_data['Latency'], 50)
            if f'{SYS_NAME}' in group:
                sw_min = min(sw_min, p50)
            elif 'GKE' in group:
                gke_max = max(gke_max, p50)
            else:
                single_lb_max = max(single_lb_max, p50)
            sns.boxplot(
                data=group_data, x='Group', y='Latency',
                whis=(10, 90),
                showfliers=False,
                showmeans=True,
                meanprops=meanprops,
                widths=0.7,
                patch_artist=True,
                boxprops=dict(facecolor='white', edgecolor=palette[i]),
                medianprops=medianprops,
                ax=ax
            )
        plt.ylabel(label)
        if do_log_scale:
            plt.yscale('log')  # Set y-axis to log scale
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Remove x-axis label (Group)
        plt.xlabel('')
        
        # Use bbox_inches='tight' instead of tight_layout to avoid white space at the bottom
        plt.savefig(f'{paper_fig_dir}/{_get_figure_name(suffix)}.pdf', bbox_inches='tight')
        plt.close()

    _plot_latency_box(ttfts, 'TTFT (s)', 'ttft')
    _plot_latency_box(e2e_latencies, 'E2E Latency (s)', 'e2e_latency', do_log_scale=False)

def eval():
    InitMatplotlib(8, 7)
    idx2group = {
        0: {
            'arena_' + k: v for k, v in desc2name.items()
        },
        1: {
            'wildchat_' + k: v for k, v in desc2name.items()
        },
        2: {
            'tot_' + k: v for k, v in desc2name.items()
        },
        3: {
            'mixed_tree_' + k: v for k, v in desc2name.items()
        },
    }
    for idx, gn2alias in idx2group.items():
        eval_plot_one(idx, gn2alias)


def ablation_sel_pushing():
    InitMatplotlib(10, 7)
    name2group = {
        'ablation_sel_pushing': {
            'ablation_sgl_bp': 'BP',
            'ablation_sgl_sp_o': 'SP-O',
            'ablation_sgl_sp_p': 'SP-P',
        },
    }
    for _, gn2alias in name2group.items():
        eval_plot_one(-1, gn2alias, fig_width_ratio=0.6)

def ablation_cross_region_tpt():
    InitMatplotlib(10, 7)
    skywalk_name = f'Sky{SYS_NAME}'
    no_cross_region_name = 'Region-Local'
    nr2groups = {
        num_replicas: {
            f'ablation_{num_replicas}_replicas_isolated': 'NoCrossRegion',
            f'ablation_{num_replicas}_replicas_skywalker_prefix': 'SkyWalker/Prefix',
        }
        for num_replicas in [18, 15, 12, 9, 6, 3]
    }
    
    # Use a nicer color palette
    palette = sns.color_palette('colorblind', 2)
    
    # Collect data for the line plot
    x_values = sorted(nr2groups.keys())  # Number of replicas
    y_values_no_cross = []
    y_values_sky_walk = []
    
    for nr in x_values:
        for exp_name, label in nr2groups[nr].items():
            metrics = get_metric_for_group(exp_name)[0]
            tpt = _get_tpt(metrics)
            
            if 'NoCrossRegion' in label:
                y_values_no_cross.append(tpt)
            elif 'SkyWalker/Prefix' in label:
                y_values_sky_walk.append(tpt)
    
    # Create the line plot with seaborn
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    # Create a DataFrame for seaborn
    data = []
    for i, nr in enumerate(x_values):
        data.append({'Replicas': nr, 'Throughput': y_values_sky_walk[i], 'Method': skywalk_name})
        data.append({'Replicas': nr, 'Throughput': y_values_no_cross[i], 'Method': no_cross_region_name})
    
    df = pd.DataFrame(data)
    
    # Calculate performance difference ratio (SkyWalker/Prefix vs NoCrossRegion)
    ratio_data = []
    for nr in x_values:
        sky_walk_tpt = df[(df['Replicas'] == nr) & (df['Method'] == skywalk_name)]['Throughput'].values[0]
        no_cross_tpt = df[(df['Replicas'] == nr) & (df['Method'] == no_cross_region_name)]['Throughput'].values[0]
        ratio = sky_walk_tpt / no_cross_tpt
        ratio_data.append({'Replicas': nr, 'Improvement Ratio': ratio})
    
    # Plot with seaborn
    # Scale down the throughput values for better display
    y_scale = 1000  # Scale factor for y-axis (10^3)
    df['Throughput_Scaled'] = df['Throughput'] / y_scale
    
    ax = sns.lineplot(
        data=df,
        x='Replicas',
        y='Throughput_Scaled',
        hue='Method',
        style='Method',
        markers=['o', 's'],
        dashes=False,
        palette=palette,
        linewidth=1.5,
        markersize=7
    )
    
    # Enhance the plot
    plt.xlabel('Number of Replicas')
    plt.ylabel('Throughput (token/s)')
    plt.xticks(x_values)
    
    # Add a subtle grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add the x10^3 label to the top of y-axis
    ax.text(0, ax.get_ylim()[1] * 0.95, '$\\times 10^3$', va='bottom', ha='left', 
            transform=ax.get_yaxis_transform())
    
    # Add arrow from isolated 12 replica point to skywalker's 9 replica point
    # Get the data points for the arrow
    isolated_12_y = df[(df['Replicas'] == 12) & (df['Method'] == no_cross_region_name)]['Throughput_Scaled'].values[0]
    skywalker_9_y = df[(df['Replicas'] == 9) & (df['Method'] == skywalk_name)]['Throughput_Scaled'].values[0]
    
    # Add the arrow
    plt.annotate('', 
                xy=(9, skywalker_9_y),      # end point (skywalker 9 replicas)
                xytext=(12, isolated_12_y), # start point (isolated 12 replicas)
                arrowprops=dict(color='#4CAF50', shrink=0.001, width=.5, headwidth=5),
                )
    
    # Add the text label
    plt.text(11, (skywalker_9_y + isolated_12_y) / 2 - 0.65, '-$25\%$', 
             color='#4CAF50', fontsize=10, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='#4CAF50', boxstyle='round,pad=0.2'))
    
    # Customize legend
    plt.legend(title=None, frameon=True, framealpha=0.9, edgecolor='lightgray', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{paper_fig_dir}/fig-10.pdf', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    eval()
    ablation_sel_pushing()
    ablation_cross_region_tpt()
