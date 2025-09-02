import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from traffic import calculate_traffic

from plot_utils import *

uid2alias = {
    'Michigan': 'us-east-1',
    'California': 'us-west',
    'Paris': 'eu-west',
    'Baden-Wurttemberg': 'eu-central',
    'New York': 'us-east-2',
}

def draw_traffic_graphs(did, user_ids, all_traffics, id2traffic):
    plt.figure(figsize=(fig_width, fig_height), dpi=300)

    hours = sorted([int(h) for h in all_traffics.keys()])
    counts = [all_traffics[str(h)] for h in hours]

    hours = sorted([int(h) for h in all_traffics.keys()])
    bottom = np.zeros(len(hours))
    palette = sns.color_palette('colorblind', len(user_ids))
    legend_elements = []
    ratios = []
    tot_hours = [0] * len(hours)
    separate_cost = 0

    for i, user_id in enumerate(user_ids):
        traffic = id2traffic[user_id]
        counts = [traffic.get(str(h), 0) for h in hours]
        ratios.append(max(counts) / min(counts))
        separate_cost += len(counts) * max(counts)
        tot_hours = [tot_hours[j] + counts[j] for j in range(len(hours))]
        plt.bar(
            hours,
            counts,
            width=0.8,
            bottom=bottom,
            align='center',
            # label=f'User {user_id}',
            color=palette[i],
        )
        bottom += np.array(counts)
        # Create smaller patch for legend
        legend_elements.append(
            Patch(facecolor='White', 
                  edgecolor=palette[i], 
                  label=uid2alias[user_id],
                  # Reduce the patch size
                  linewidth=0.8))

    print('aggregated ratio', max(tot_hours) / min(tot_hours))
    print('ratios', ratios)
    print('separate cost', separate_cost)
    print('total cost', max(tot_hours) * len(hours))
    print('od cost', sum(tot_hours) / 0.4)

    plt.xticks(hours)
    plt.xlabel('Hour of the Day')
    plt.ylabel('#Requests')
    # plt.title('Aggregated Traffic',fontdict={'fontsize': font_size})
    y_scale = 1000
    ymax = plt.gca().get_ylim()[1]
    print(ymax)
    # Scale down the y-axis values by 1000
    scaled_ymax = int(ymax / y_scale) + 1
    yticks = list(range(0, scaled_ymax + 1))
    print(yticks)
    plt.yticks([y * y_scale for y in yticks], yticks)
    plt.text(-3, ymax*0.95, '$\\times 10^3$', va='bottom', ha='center')
    plt.ylim(0, ymax)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    x_scale = 1
    # plt.xlim(0, x_cutoff)
    x_ticks = list(range(0, 24 // x_scale, 3))
    plt.xticks([x * x_scale for x in x_ticks], x_ticks)
    # Adjust legend to use smaller patches and spacing
    plt.legend(handles=legend_elements, 
               loc='lower center', 
               bbox_to_anchor=(0.55, .9),
               ncol=3,
               fontsize=7)  # Reduce space between columns

    plt.tight_layout()
    plt.savefig(f'{paper_fig_dir}/fig-3-a.pdf', bbox_inches='tight')
    plt.close()

def main(
    did,
    user_ids,
    average_traffic=False,
):
    all_traffics = defaultdict(int)
    id2traffic = {}
    with open(f"dataset_cache/{did}/userid2fn.json", "r") as f:
        userid2fn = json.load(f)
    # First calculate max value across all users
    max_value = 0
    if average_traffic:
        for user_id in user_ids:
            fn = userid2fn[str(user_id)]
            with open(fn, "r") as f:
                data = json.load(f)
            max_value = max(max_value, max(data.values()))

    # Then normalize using the global max
    for user_id in user_ids:
        fn = userid2fn[str(user_id)]
        with open(fn, "r") as f:
            data = json.load(f)
            if average_traffic:
                # Average the traffic across all users
                data = {
                    hour: count * max_value / max(data.values())
                    for hour, count in data.items()
                }
            id2traffic[user_id] = data
            for hour, count in data.items():
                all_traffics[hour] += count
    draw_traffic_graphs(did, user_ids, all_traffics, id2traffic)

if __name__ == "__main__":
    InitMatplotlib(10,7)
    calculate_traffic("wildchat-state", force_reload=True, num_top=12)
    main("wildchat-state", ['Michigan', 'California', 'Paris', 'Baden-Wurttemberg', 'New York'])

