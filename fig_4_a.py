import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import datasets
import json
from plot_utils import *
from tqdm import tqdm

colors = ['#0072B2', '#009E73']

def _get_conv_i(d, i):
    c = d['conversation'][i]
    return {'role': c['role'], 'content': c['content']}

def _get_input_output_lens(dataset_name, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        input_lens = data['input_lens']
        output_lens = data['output_lens']
    else:
        input_lens = []
        output_lens = []
        ds = datasets.load_dataset(
            dataset_name,
            split="train",
            num_proc=os.cpu_count(),
            columns=["conversation", 'turn'],
        )
        for d in tqdm(ds):
            history = []
            for i in range(d['turn']):
                history.append(_get_conv_i(d, i * 2))
                input_len = len(json.dumps(history))
                input_lens.append(input_len)
                history.append(_get_conv_i(d, i * 2 + 1))
                output_len = len(json.dumps(history))
                output_lens.append(output_len)
        with open(cache_file, 'w') as f:
            json.dump({'input_lens': input_lens, 'output_lens': output_lens}, f)
    return input_lens, output_lens


def _get_cache_file(dataset_name):
    return f'dataset_cache/{dataset_name.replace("/", "_")}_len_distribution.json'


def plot_cdf(dataset_name='allenai/WildChat-1M'):
    input_lens, output_lens = _get_input_output_lens(dataset_name, _get_cache_file(dataset_name))
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    legend_elements = []
    PlotCdf(input_lens, label='Input', color=colors[0])
    legend_elements.append(Patch(facecolor='White', edgecolor=colors[0], label='Input'))
    PlotCdf(output_lens, label='Output', color=colors[1])
    legend_elements.append(Patch(facecolor='White', edgecolor=colors[1], label='Output'))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlabel('Length')
    plt.ylabel('CDF')
    plt.xlim(0, 11000)
    plt.legend(handles=legend_elements, loc='upper left', facecolor='white', 
               frameon=True, edgecolor='lightgray', framealpha=0.9, 
               borderpad=0.5, bbox_to_anchor=(0, 1.05))
    # plt.tight_layout()
    plt.savefig(f'{paper_fig_dir}/fig-4-a.pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    InitMatplotlib(11, 7)
    plot_cdf()