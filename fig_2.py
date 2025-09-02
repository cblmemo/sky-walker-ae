import matplotlib.pyplot as plt
from traffic import load_raw_dataset
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone


from plot_utils import *

def plot_diurnal_pattern(did, num_figure, n_cols=3):
    print(f"Calculating traffic for {did}")
    user_data = load_raw_dataset(did, force_reload=False)

    user_data_raw = {user: timestamps for user, timestamps in user_data.items()}
    top_n = sorted(
        user_data_raw.keys(), key=lambda x: len(user_data_raw[x]), reverse=True
    )
    top_n = top_n[:num_figure]
    print([(u, len(user_data_raw[u])) for u in top_n])
    user_data = {
        user: timestamps for user, timestamps in user_data.items() if user in top_n
    }

    n_users = len(top_n)
    n_rows = (n_users + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("colorblind", n_colors=n_users)
    axes = axes.flatten()

    for i, user in enumerate(top_n):
        timestamps = user_data[user]
        # pst = timezone(timedelta(hours=-8))
        utc = timezone.utc
        times_of_day = [datetime.fromtimestamp(ts, tz=utc).time() for ts in timestamps]
        hour_counts = Counter(time.hour for time in times_of_day)
        hours = sorted(hour_counts.keys())
        counts = [hour_counts[hour] for hour in hours]

        ax = axes[i]
        # Create white bars
        ax.bar(hours, counts, width=0.6, align="center", edgecolor='gray', color='white', alpha=0.6)
        
        # Add colored smooth curve on top of the bars
        # Fill in missing hours with zeros to make a smooth curve
        all_hours = list(range(24))
        all_counts = [hour_counts.get(h, 0) for h in all_hours]
        ax.plot(all_hours, all_counts, '-', linewidth=1, color=palette[i])
        
        ax.set_xticks(range(0, 24, 4))
        ymax = int(ax.get_ylim()[1])+1
        step = 500 if ymax < 5000 else 2000
        ax.set_yticks(range(0, ymax, step))
        ax.set_xlim(-0.5, 23.5)
        if (i // n_cols) % n_rows == n_rows - 1:
            ax.set_xlabel("Hour of the Day")
        if i % n_cols == 0:
            ax.set_ylabel("Number of Requests")
        ax.set_title(user, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{paper_fig_dir}/fig-2.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    InitMatplotlib(11, 7)
    plot_diurnal_pattern("wildchat-country", 6)
