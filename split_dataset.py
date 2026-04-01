"""
Split data_ece dataset into train/val/test splits at the VIDEO level.

Sub-chunks of the same video (e.g., chunk_024_00, chunk_024_01, ...) are kept
together in the same split to avoid temporal data leakage.

Target ratio: ~70% train, ~15% val, ~15% test (by frame count).
Saves split info as JSON (no file copying/moving).
"""

import os
import re
import json
import random

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_ece")
SEED = 42


def get_video_info():
    """Group chunk dirs by video ID and count frames."""
    chunks = sorted([
        d for d in os.listdir(BASE_DIR)
        if d.startswith("chunk_") and os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    videos = {}
    for c in chunks:
        m = re.match(r"chunk_(\d+)", c)
        vid_id = int(m.group(1))
        n_frames = len([f for f in os.listdir(os.path.join(BASE_DIR, c)) if f.endswith(".npz")])
        if vid_id not in videos:
            videos[vid_id] = {"chunks": [], "total_frames": 0}
        videos[vid_id]["chunks"].append(c)
        videos[vid_id]["total_frames"] += n_frames

    return videos


def greedy_split(videos, train_ratio=0.70, val_ratio=0.15):
    """
    Greedy bin-packing split to match target ratios as closely as possible.
    Sort videos by frame count descending, assign each to the split that is
    furthest below its target.
    """
    total_frames = sum(v["total_frames"] for v in videos.values())
    target_train = train_ratio * total_frames
    target_val = val_ratio * total_frames
    target_test = (1 - train_ratio - val_ratio) * total_frames

    # Shuffle then sort by frame count (descending) for deterministic greedy packing
    vid_ids = list(videos.keys())
    random.seed(SEED)
    random.shuffle(vid_ids)
    vid_ids.sort(key=lambda v: videos[v]["total_frames"], reverse=True)

    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    for vid_id in vid_ids:
        n = videos[vid_id]["total_frames"]
        # Assign to the split with the largest remaining deficit
        deficits = {s: targets[s] - counts[s] for s in splits}
        best_split = max(deficits, key=deficits.get)
        splits[best_split].append(vid_id)
        counts[best_split] += n

    return splits, counts, total_frames


def main():
    videos = get_video_info()
    total_frames = sum(v["total_frames"] for v in videos.values())

    print(f"Dataset: {len(videos)} videos, {total_frames} total frames\n")

    splits, counts, _ = greedy_split(videos)

    # Build output
    result = {}
    for split_name in ["train", "val", "test"]:
        vid_ids = sorted(splits[split_name])
        split_chunks = []
        for vid_id in vid_ids:
            split_chunks.extend(videos[vid_id]["chunks"])

        result[split_name] = {
            "video_ids": vid_ids,
            "chunks": split_chunks,
            "n_videos": len(vid_ids),
            "n_chunks": len(split_chunks),
            "n_frames": counts[split_name],
            "pct": round(counts[split_name] / total_frames * 100, 1),
        }

    # Print summary
    print(f"{'Split':<8} {'Videos':<8} {'Chunks':<8} {'Frames':<8} {'%':<6}")
    print("-" * 38)
    for split_name in ["train", "val", "test"]:
        r = result[split_name]
        print(f"{split_name:<8} {r['n_videos']:<8} {r['n_chunks']:<8} {r['n_frames']:<8} {r['pct']:<6}")

    print()
    for split_name in ["train", "val", "test"]:
        r = result[split_name]
        print(f"{split_name}: videos {r['video_ids']}")
        print(f"  chunks: {r['chunks']}\n")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_ece_split.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Split saved to: {out_path}")


if __name__ == "__main__":
    main()
