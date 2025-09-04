import json
import os
import random
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import partial
from itertools import combinations
from multiprocessing import Pool

import datasets
import matplotlib.pyplot as plt
import numpy as np
import requests
import tqdm
from rich import print
from torch.utils.data import DataLoader

id2dataset = {
    "chat-conv": "lmsys/chatbot_arena_conversations",
    "vision-chat": "lmarena-ai/VisionArena-Chat",
    "vision-battle": "lmarena-ai/VisionArena-Battle",
    "arena-all": "dummy-name",
    "arena-all-language": "dummy-name",
    "wildchat": "allenai/WildChat-1M",
    "wildchat-country": "allenai/WildChat-1M",
    "wildchat-state": "allenai/WildChat-1M",
}


def download_in_threads(url, num_threads):
    def fetch_range(url, start, end, results, index):
        headers = {"Range": f"bytes={start}-{end}"}
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 206:
            results[index] = response.content
        else:
            results[index] = b""

    response = requests.head(url)
    if "Content-Length" not in response.headers:
        raise ValueError(
            "Server does not provide Content-Length, cannot split the file."
        )
    if (
        "Accept-Ranges" not in response.headers
        or response.headers["Accept-Ranges"] != "bytes"
    ):
        raise ValueError("Server does not support range requests.")
    print(response.headers)

    content_length = int(response.headers["Content-Length"])
    chunk_size = content_length // num_threads

    threads = []
    results = [None] * num_threads

    for i in range(num_threads):
        start = i * chunk_size
        end = content_length - 1 if i == num_threads - 1 else (start + chunk_size - 1)
        thread = threading.Thread(
            target=fetch_range, args=(url, start, end, results, i)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return b"".join(results)


def parallel_fetch_url(url, fn):
    content = download_in_threads(url, num_threads=os.cpu_count())
    with open(fn, "wb") as f:
        f.write(content)
    print("Download complete!")


def process_batch_vision(batch, userid_field_name):
    uid_timestamps = {}
    for data in batch:
        user_id = str(data[userid_field_name])
        tstamp = float(data["tstamp"])
        if user_id not in uid_timestamps:
            uid_timestamps[user_id] = []
        uid_timestamps[user_id].append(tstamp)
    return uid_timestamps


process_batch_vision_chat = partial(process_batch_vision, userid_field_name="user_id")
process_batch_vision_battle = partial(process_batch_vision, userid_field_name="judge")


def load_raw_dataset(did, force_reload=False):
    os.makedirs("dataset_cache", exist_ok=True)
    cache_file_name = f"dataset_cache/{did}.json"
    if os.path.exists(cache_file_name) and not force_reload:
        with open(cache_file_name, "r") as f:
            uid_timestamps = json.load(f)
        return uid_timestamps

    uid_timestamps = defaultdict(list)

    if did == "chat-conv":
        ds = datasets.load_dataset(id2dataset[did], split="train")
        for data in ds:
            user = data["judge"]
            uid_timestamps[user].append(data["tstamp"])
    elif did in ["vision-chat", "vision-battle"]:
        # Estimated: 10 mins for chat, 3 mins for battle
        batch_size = 100
        if did == "vision-chat":
            collate_fn = process_batch_vision_chat
            user_id_field_name = "user_id"
            total = 2000
        else:
            collate_fn = process_batch_vision_battle
            user_id_field_name = "judge"
            total = 300
        ds = datasets.load_dataset(
            id2dataset[did],
            split="train",
            # https://github.com/huggingface/datasets/issues/4114#issuecomment-1956450114
            streaming=True,
            columns=[user_id_field_name, "tstamp"],
        )
        if did == "vision-battle":
            # Vision battle does not works with the dataloader for some reason.
            for data in tqdm.tqdm(ds, total=total * batch_size):
                user_id = str(data[user_id_field_name])
                tstamp = float(data["tstamp"])
                uid_timestamps[user_id].append(tstamp)
        else:
            ds = ds.with_format("torch")
            dataloader = DataLoader(
                ds,
                batch_size=100,
                num_workers=os.cpu_count(),
                collate_fn=collate_fn,
            )
            for batch_timestamps in tqdm.tqdm(dataloader, total=total):
                for user_id, timestamps in batch_timestamps.items():
                    uid_timestamps[user_id].extend(timestamps)
    elif did.startswith("arena-all"):
        raw_file_fn = "dataset_cache/arena_all_raw.json"
        if not os.path.exists(raw_file_fn):
            # https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH
            arena_all_url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json"
            parallel_fetch_url(arena_all_url, raw_file_fn)
        with open(raw_file_fn, "r") as file:
            data = json.load(file)
        for d in data:
            if did.endswith("-language"):
                user = d["language"]
                # Skip English and unknown as this two has global traffic.
                if user in [
                    # "English",
                    "unknown",
                ]:
                    continue
            else:
                user = d["judge"]
            uid_timestamps[user].append(d["tstamp"])
    elif did.startswith("wildchat"):
        if did.endswith("-country"):
            user_field_name = "country"
        elif did.endswith("-state"):
            user_field_name = "state"
        else:
            user_field_name = "hashed_ip"

        ds = datasets.load_dataset(
            id2dataset[did],
            split="train",
            num_proc=os.cpu_count(),
            columns=[user_field_name, "timestamp"],
        )

        def process_row(data):
            user = data[user_field_name]
            if user is None:
                return {"user": None, "tstamp": None}
            tstamp = data["timestamp"].timestamp()
            return {"user": user, "tstamp": tstamp}

        results = ds.map(
            process_row, num_proc=os.cpu_count(), remove_columns=ds.column_names
        )

        for row in tqdm.tqdm(results, desc=f"Processing {did}"):
            if row["user"] is not None:
                uid_timestamps[row["user"]].append(row["tstamp"])
    else:
        raise ValueError(f"Unknown dataset ID: {did}")

    with open(cache_file_name, "w") as f:
        json.dump(uid_timestamps, f, indent=2)

    return uid_timestamps


def calculate_traffic(did, force_reload=False, plot=False, num_top=3):
    n_cols = 3

    if not plot and not force_reload and os.path.exists(f"dataset_cache/{did}/userid2fn.json"):
        with open(f"dataset_cache/{did}/userid2fn.json", "r") as f:
            userid2fn = json.load(f)
        if len(userid2fn) == num_top:
            print(f"Skipping {did} because it has already been calculated")
            return

    print(f"Calculating traffic for {did}")
    user_data = load_raw_dataset(did, force_reload=force_reload and not plot)

    user_data_raw = {user: timestamps for user, timestamps in user_data.items()}
    top_n = sorted(
        user_data_raw.keys(), key=lambda x: len(user_data_raw[x]), reverse=True
    )
    top_n = [i for i in top_n if i != 'Hong Kong']
    top_n = top_n[:num_top]
    print([(u, len(user_data_raw[u])) for u in top_n])
    user_data = {
        user: timestamps for user, timestamps in user_data.items() if user in top_n
    }

    os.system(f"rm -rf dataset_cache/{did}")
    os.makedirs(f"dataset_cache/{did}", exist_ok=True)

    userid2fn = {}

    for i, user in enumerate(top_n):
        timestamps = user_data[user]
        # pst = timezone(timedelta(hours=-8))
        utc = timezone.utc
        times_of_day = [datetime.fromtimestamp(ts, tz=utc).time() for ts in timestamps]
        hour_counts = Counter(time.hour for time in times_of_day)
        hours = sorted(hour_counts.keys())
        counts = [hour_counts[hour] for hour in hours]
        source_data = {str(hour): count for hour, count in zip(hours, counts)}
        user_index = str(i).zfill(3)
        fn = f"dataset_cache/{did}/traffic_{user_index}_{user}.json"
        with open(fn, "w") as f:
            json.dump(source_data, f, indent=2)
        userid2fn[user] = fn

    with open(f"dataset_cache/{did}/userid2fn.json", "w") as f:
        json.dump(userid2fn, f, indent=2)
