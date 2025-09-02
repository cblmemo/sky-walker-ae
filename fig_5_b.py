import pandas as pd
from itertools import combinations, product
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import datasets
import time
import json
import multiprocessing as mp
from tqdm import tqdm
import argparse
import os
from functools import partial
import seaborn as sns
from plot_utils import *

DATASET_NAME = "allenai/WildChat-1M"
CONV_SELECTOR = 'conversation_a'
DATASET_NAME_1M = 'lmsys/lmsys-chat-1m'


def _extract(d: Dict[str, Any], user_field: str = 'judge') -> Dict[str, Any]:
    return {
        # 'turn': d['turn'],
        # 'tstamp': d['tstamp'],
        'user': d[user_field],
        'conv': d[CONV_SELECTOR],
    }


def _load_arena_dataset() -> List[Dict[str, Any]]:
    tic = time.time()
    num_cores = mp.cpu_count()
    multi_turn_data = []
    chunk_data = datasets.load_dataset('lmsys/chatbot_arena_conversations', split='train', num_proc=num_cores)
    for d in tqdm(chunk_data, desc='Loading dataset', unit='conversations'):
        if d['turn'] > 1:
            multi_turn_data.append(_extract(d))
    print(f'Got {len(multi_turn_data)} multi-turn conversations '
          f'(took {time.time() - tic:.2f}s)')
    return multi_turn_data


def _load_arena_dataset_1m(sample_size: int = 100000, user_field: str = 'user') -> List[Dict[str, Any]]:
    tic = time.time()
    num_cores = mp.cpu_count()
    slice_str = f'[:{sample_size}]'
    chunk_data = datasets.load_dataset(DATASET_NAME_1M,
                                       split=f'train{slice_str}',
                                       num_proc=num_cores)
    multi_turn_data = []
    for d in tqdm(chunk_data, desc='Loading dataset', unit='conversations'):
        # from rich import print
        # print(d)
        if d['turn'] > 1:
            multi_turn_data.append({'conv': d['conversation'], 'user': d[user_field]})
    print(f'Got {len(multi_turn_data)} multi-turn conversations '
          f'(took {time.time() - tic:.2f}s)')
    return multi_turn_data

# _load_arena_dataset_1m(1)
# exit()


def _load_dataset(sample_size: int = 1000, user_field: str = 'user') -> List[Dict[str, Any]]:
    tic = time.time()
    num_cores = mp.cpu_count()
    sample_slice = ''
    if sample_size is not None:
        sample_slice = f'[:{sample_size}]'
    chunk_data = datasets.load_dataset(DATASET_NAME, split=f'train{sample_slice}', num_proc=num_cores)
    # chunk_data = datasets.load_dataset(DATASET_NAME, split='train')
    multi_turn_data = []
    for d in tqdm(chunk_data, desc='Loading dataset', unit='conversations'):
        # At least 2 full turns: user + assistant + user + assistant (len >= 4)
        if d['turn'] >= 2 and isinstance(d['conversation'],
                                         list) and len(d['conversation']) >= 4:
            if d.get(user_field, 'unknown') == 'unknown':
                continue
            # if not _filter_conv_by_region(d, region):
            #     continue
            raw_conv = []
            for c in d['conversation']:
                raw_conv.append({'role': c['role'], 'content': c['content']})
            conv = {
                # 'turn': d['turn'],
                # 'timestamp': d['timestamp'].timestamp(),
                'conv': raw_conv,
                'user': d[user_field],
                # 'state': d['state'],
                # 'country': d['country'],
            }
            multi_turn_data.append(conv)
    print(f'Got {len(multi_turn_data)} multi-turn '
          f'conversations (took {time.time() - tic:.2f}s)')
    # random.shuffle(multi_turn_data)
    return multi_turn_data


def _prefix_sim(a: str, b: str) -> float:
    """Fraction of the *shorter* string that forms a common prefix."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / n                            # ∈ [0, 1]

def _filter_conv(conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out non-user messages and keep only user messages."""
    return [{'role': m['role'], 'content': m['content']} for m in conv]


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def build_request_df(dataset: str = DATASET_NAME, sample_size: int = 1000, user_field: str = 'user') -> pd.DataFrame:
    """Load → filter → flatten into 〈ip, text〉 rows."""
    if dataset == DATASET_NAME:
        raw: List[Dict[str, Any]] = _load_dataset(sample_size, user_field)
    elif dataset == DATASET_NAME_1M:
        raw: List[Dict[str, Any]] = _load_arena_dataset_1m(sample_size, user_field)
    else:
        raw: List[Dict[str, Any]] = _load_arena_dataset()
    rows = []
    for r in tqdm(raw, desc='Building request dataframe', unit='records'):
        for i in range(0, len(r['conv']), 2):
            rows.append({'ip': r['user'], 'text': json.dumps(_filter_conv(r['conv'][:i+2]))})
    df = pd.DataFrame(rows)
    # print(df)
    if df.empty:
        raise ValueError('No valid rows found')
    df = df[df['text'].str.len() > 0]       # drop empty / corrupt rows
    unique_ips = df['ip'].nunique()
    print(f'Number of unique IPs: {unique_ips}')
    return df

def _calc_within_user_sim(texts, ip=None):
    """Calculate similarity scores for combinations of texts from the same user."""
    scores = []
    for a, b in combinations(texts, 2):
        scores.append(_prefix_sim(a, b))
    return scores, ip

def _calc_cross_user_sim_batch(pairs_batch):
    """Calculate similarity scores for a batch of cross-user text pairs."""
    scores = []
    for a, b, ip_a, ip_b in pairs_batch:
        scores.append((_prefix_sim(a, b), ip_a, ip_b))
    return scores

def _create_cross_user_batches(user_texts, batch_size=1000):
    """Create batches of cross-user text pairs for parallel processing."""
    all_pairs = []
    for (ip_a, texts_a), (ip_b, texts_b) in tqdm(
            list(combinations(user_texts.items(), 2)),
            desc='Creating cross-user pairs',
            unit='pairs'):
        if ip_a == ip_b:
            continue
        for a in texts_a:
            for b in texts_b:
                all_pairs.append((a, b, ip_a, ip_b))
    
    # Split into batches
    batches = []
    for i in tqdm(range(0, len(all_pairs), batch_size),
                 desc='Creating batches',
                 unit='batch'):
        batches.append(all_pairs[i:i + batch_size])
    
    return batches

# ------------------------------------------------------------------ fast CDF
def fast_cdf(scores_iterable, bins=2048, chunk=1_000_000):
    """
    Build an approximate CDF in a single pass without sorting.
    - scores_iterable: any iterable (list, np.ndarray, generator, …)
    - bins:            number of histogram buckets (CDF resolution)
    - chunk:           how many elements to load per batch (controls RAM)
    Returns (x, cdf) ready for a matplotlib step plot.
    """
    edges   = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)      # fixed bin edges
    counts  = np.zeros(bins,        dtype=np.int64)

    buf = []
    for v in scores_iterable:
        buf.append(v)
        if len(buf) == chunk:
            idx = np.minimum((np.asanyarray(buf) * bins).astype(np.int64), bins - 1)
            counts += np.bincount(idx, minlength=bins)
            buf.clear()
    # flush leftovers
    if buf:
        idx = np.minimum((np.asanyarray(buf) * bins).astype(np.int64), bins - 1)
        counts += np.bincount(idx, minlength=bins)

    cdf = np.cumsum(counts) / counts.sum()
    return edges[1:], cdf                      # drop leftmost edge for plotting

def plot_cdf_fast(scores, out_path, color, label, bins=2048):
    x, y = fast_cdf(scores, bins=bins)
    plt.step(x, y, where='post', lw=1.2, label=label, color=color)

# --- CDF plot ---------------------------------------------------------------
def _plot_cdf(within, cross, out_dir):
    print('Generating CDF plot...')
    plt.figure(figsize=(4, 3))
    plot_cdf_fast(within_user_scores, 'dummy', '#1f77b4', 'within')
    plot_cdf_fast(cross_user_scores,  'dummy', '#d62728', 'cross')
    plt.xlabel('prefix-similarity')
    plt.ylabel('fraction ≤ x')
    plt.legend(frameon=False)
    plt.tight_layout()
    print(f'Saving CDF plot to {out_dir}/similarity_cdf.pdf')
    plt.savefig(f'{out_dir}/similarity_cdf_fast.pdf', dpi=300)
    plt.close()
    print('CDF plot generation complete')

def analyze_similarity_distribution(dataset: str = DATASET_NAME, sample_size: int = 1000, user_field: str = 'user', output_dir: str = 'figs') -> Tuple[Dict, Dict, List, List, Dict]:
    """Analyze similarity distribution within and across users."""
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if results already exist
    results_file = f'{output_dir}/similarity_results.json'
    within_scores_file = f'{output_dir}/within_user_scores.npy'
    cross_scores_file = f'{output_dir}/cross_user_scores.npy'
    user_scores_file = f'{output_dir}/user_similarity_scores.json'
    cross_user_pairs_file = f'{output_dir}/cross_user_pairs.json'
    
    if (os.path.exists(results_file) and 
        os.path.exists(within_scores_file) and 
        os.path.exists(cross_scores_file) and
        os.path.exists(user_scores_file) and
        os.path.exists(cross_user_pairs_file)):
        print(f'Found existing results in {output_dir}, loading data...')
        
        # Load existing results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        with open(user_scores_file, 'r') as f:
            user_similarity_scores = json.load(f)
            
        with open(cross_user_pairs_file, 'r') as f:
            cross_user_pairs = json.load(f)
        
        within_user_scores = np.load(within_scores_file).tolist()
        cross_user_scores = np.load(cross_scores_file).tolist()
        
        within_stats = results['within_user']
        cross_stats = results['cross_user']
        
        print(f'Loaded existing similarity results from {output_dir}')
        return within_stats, cross_stats, within_user_scores, cross_user_scores, user_similarity_scores
    
    # If results don't exist, compute them
    print('No existing results found, computing similarity distributions...')
    tic = time.time()

    print(f'Loading dataset with sample size {sample_size} '
          f'and user field {user_field}')
    df = build_request_df(dataset, sample_size, user_field)
    
    # Within user similarity - prepare groups
    user_groups = []
    user_ips = []
    user_similarity_scores = {}
    
    for ip, g in df.groupby('ip'):
        texts = g['text'].tolist()
        user_groups.append(texts)
        user_ips.append(ip)
        
        # Calculate and store similarity scores for each user
        if len(texts) >= 2:  # Need at least 2 texts to calculate similarity
            scores = []
            for a, b in combinations(texts, 2):
                score = _prefix_sim(a, b)
                scores.append(score)
            
            if scores:  # Only store if we have scores
                user_similarity_scores[ip] = {
                    'mean_similarity': np.mean(scores),
                    'num_texts': len(texts),
                    'scores': scores
                }
    
    # Cross user similarity - prepare user texts
    user_texts = {}
    for ip, g in df.groupby('ip'):
        user_texts[ip] = g['text'].tolist()
    
    # Parallel processing
    num_cores = mp.cpu_count()
    print(f'Using {num_cores} CPU cores for parallel processing')
    
    # Process within-user similarities
    print('Calculating within-user similarities...')
    # Split large user groups into smaller chunks for better load balancing
    chunked_user_groups = []
    chunked_user_ips = []
    for group, ip in zip(user_groups, user_ips):
        if len(group) > 100:  # If a user has many texts, split into chunks
            for i in range(0, len(group), 100):
                chunked_user_groups.append(group[i:i+100])
                chunked_user_ips.append(ip)
        else:
            chunked_user_groups.append(group)
            chunked_user_ips.append(ip)
    
    with mp.Pool(processes=num_cores) as pool:
        within_results = list(tqdm(
            pool.starmap(_calc_within_user_sim, 
                         [(group, ip) for group, ip in zip(chunked_user_groups, chunked_user_ips)], 
                         chunksize=max(1, len(chunked_user_groups)//num_cores//4)),
            total=len(chunked_user_groups),
            desc='Within-user similarity',
            unit='groups'
        ))
    
    within_user_scores = []
    for scores, ip in within_results:
        within_user_scores.extend(scores)
    
    # Process cross-user similarities with batching for better load balancing
    print('Calculating cross-user similarities...')
    cross_batches = _create_cross_user_batches(user_texts, batch_size=5000)
    print(f'Created {len(cross_batches)} batches for cross-user similarity calculation')
    
    with mp.Pool(processes=num_cores) as pool:
        cross_results = list(tqdm(
            pool.imap(_calc_cross_user_sim_batch, cross_batches, chunksize=max(1, len(cross_batches)//num_cores//4)),
            total=len(cross_batches),
            desc='Cross-user similarity',
            unit='batches'
        ))
    
    # Store cross-user similarity with user IPs
    cross_user_scores = []
    cross_user_pairs = {}
    
    for batch_results in cross_results:
        for score, ip_a, ip_b in batch_results:
            cross_user_scores.append(score)
            
            # Store cross-user similarity by user pair
            pair_key = f"{ip_a}_{ip_b}"
            if pair_key not in cross_user_pairs:
                cross_user_pairs[pair_key] = {
                    'ip_a': ip_a,
                    'ip_b': ip_b,
                    'scores': []
                }
            cross_user_pairs[pair_key]['scores'].append(score)
    
    # Calculate mean for each cross-user pair
    for pair_key, data in cross_user_pairs.items():
        data['mean_similarity'] = np.mean(data['scores'])
    
    # Save cross-user pairs data
    # with open(f'{output_dir}/cross_user_pairs.json', 'w') as f:
    #     json.dump(cross_user_pairs, f, indent=2)
    
    print(f'Similarity calculations completed in {time.time() - tic:.2f}s')
    
    # Calculate statistics
    within_stats = {
        'mean': np.mean(within_user_scores),
        'median': np.median(within_user_scores),
        'std': np.std(within_user_scores),
        'min': min(within_user_scores),
        'max': max(within_user_scores),
        'count': len(within_user_scores)
    }
    
    cross_stats = {
        'mean': np.mean(cross_user_scores),
        'median': np.median(cross_user_scores),
        'std': np.std(cross_user_scores),
        'min': min(cross_user_scores),
        'max': max(cross_user_scores),
        'count': len(cross_user_scores)
    }
    
    # Save raw scores to files
    # np.save(f'{output_dir}/within_user_scores.npy', np.array(within_user_scores))
    # np.save(f'{output_dir}/cross_user_scores.npy', np.array(cross_user_scores))
    
    # Save user similarity scores
    # with open(f'{output_dir}/user_similarity_scores.json', 'w') as f:
    #     json.dump(user_similarity_scores, f, indent=2)
    
    # Save statistics to JSON
    results = {
        'within_user': within_stats,
        'cross_user': cross_stats,
        'ratio': within_stats['mean'] / cross_stats['mean']
    }
    
    # with open(f'{output_dir}/similarity_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    
    return within_stats, cross_stats, within_user_scores, cross_user_scores, user_similarity_scores, cross_user_pairs

# --- Top users heatmap ------------------------------------------------------
def _plot_top_users_heatmap(user_similarity_scores, out_dir, top_n=10, cross_user_pairs=None):
    """Plot heatmap of similarity scores for randomly selected users."""
    InitMatplotlib(11,13)
    print(f'Generating heatmap for {top_n} randomly selected users...')
    
    if cross_user_pairs is None:
        print(f'Cross-user pairs file not found: {cross_user_pairs}')
        return
    
    # Filter users with at least 3 texts (to have meaningful similarity)
    filtered_users = {k: v for k, v in user_similarity_scores.items() if v['num_texts'] >= 3}
    
    if not filtered_users:
        print('No users with sufficient texts found for heatmap')
        return
    
    # Randomly select users instead of sorting by similarity
    import random
    user_items = list(filtered_users.items())
    random.shuffle(user_items)
    selected_users = user_items[:top_n]
    
    # Create a matrix for the heatmap
    user_ids = [user_id for user_id, _ in selected_users]
    similarity_matrix = np.zeros((top_n, top_n))
    
    # Fill the matrix with mean similarity values
    for i, (user_a, data_a) in enumerate(selected_users):
        similarity_matrix[i, i] = data_a['mean_similarity']  # Diagonal is within-user similarity
        
        # For off-diagonal elements, use the actual cross-user similarity
        for j, (user_b, _) in enumerate(selected_users):
            if i != j:
                # Look up cross-user similarity in both directions
                pair_key_1 = f"{user_a}_{user_b}"
                pair_key_2 = f"{user_b}_{user_a}"
                
                # Include results from both keys if they exist
                if pair_key_1 in cross_user_pairs and pair_key_2 in cross_user_pairs:
                    # Average the similarities from both directions
                    sim_1 = cross_user_pairs[pair_key_1]['mean_similarity']
                    sim_2 = cross_user_pairs[pair_key_2]['mean_similarity']
                    similarity_matrix[i, j] = (sim_1 + sim_2) / 2
                elif pair_key_1 in cross_user_pairs:
                    similarity_matrix[i, j] = cross_user_pairs[pair_key_1]['mean_similarity']
                elif pair_key_2 in cross_user_pairs:
                    similarity_matrix[i, j] = cross_user_pairs[pair_key_2]['mean_similarity']
                else:
                    # If no direct comparison exists, use a placeholder
                    similarity_matrix[i, j] = 0.0
    
    # Create a more readable version of user IDs for display
    # display_ids = [f"User {i+1}\n({data['mean_similarity']:.3f})" for i, (_, data) in enumerate(selected_users)]
    display_ids = [str(i) for i, (_, data) in enumerate(selected_users)]
    display_ids = ["" for i, (_, data) in enumerate(selected_users)]
    
    # Plot the heatmap
    plt.figure(figsize=(fig_width * 1.25, fig_width))
    sns.heatmap(
        similarity_matrix,
        annot=False,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=display_ids,
        yticklabels=display_ids,
        vmin=0,
        vmax=1
    )
    plt.title(f'{top_n} Randomly Selected Users')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fig-5-b.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the data for these selected users
    # selected_users_data = {user_id: data for user_id, data in selected_users}
    # with open(f'{out_dir}/random_users_similarity.json', 'w') as f:
    #     json.dump(selected_users_data, f, indent=2)
    
    print(f'Heatmap saved to {out_dir}/random_users_similarity_heatmap.pdf')
    # print(f'Random users data saved to {out_dir}/random_users_similarity.json')




# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # For multiprocessing to work correctly on Windows
    mp.freeze_support()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze similarity in conversation data')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples to load from dataset (default: 1000)')
    parser.add_argument('--user-field', type=str, default='user',
                        help='Field to use as user identifier (default: "user")')
    parser.add_argument('--output-dir', type=str, default=paper_fig_dir,
                        help='Directory to save output files (default: "figs")')
    parser.add_argument('--dataset', type=str, default=DATASET_NAME,
                        help=f'Dataset name to use (default: "{DATASET_NAME}")')
    args = parser.parse_args()
    
    # Calculate and display similarity distribution analysis
    within_stats, cross_stats, within_user_scores, cross_user_scores, user_similarity_scores, cross_user_pairs = analyze_similarity_distribution(args.dataset, args.sample_size, args.user_field, args.output_dir)
    
    print('\n=== Similarity Analysis ===')
    print('Within-User Similarity Statistics:')
    for k, v in within_stats.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    
    print('\nCross-User Similarity Statistics:')
    for k, v in cross_stats.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    
    # Calculate ratio of mean similarities
    ratio = within_stats['mean'] / cross_stats['mean']
    print(f'\nRatio of mean within-user to cross-user similarity: {ratio:.2f}x')
    print('(Higher ratio indicates users are more similar to themselves than to others)')
    print(f'\nResults saved to {args.output_dir}/similarity_results.json')
    
    # Generate plots
    # _plot_cdf(within_user_scores, cross_user_scores, args.output_dir)
    
    _plot_top_users_heatmap(user_similarity_scores, args.output_dir, 100, cross_user_pairs)
