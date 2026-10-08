"""
Preprocess SNAP "sx-*" family temporal networks (sx-askubuntu, sx-superuser,
sx-stackoverflow, sx-mathoverflow) into time-binned graph snapshots,
compatible with the same world_model.py / train_world_model.py pipeline
used for the Reddit Hyperlinks dataset.

"""

import argparse
import pickle
from collections import Counter
from datetime import datetime, timedelta, timezone

import networkx as nx


def load_edges(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                src, dst, ts = parts[0], parts[1], int(parts[2])
            except ValueError:
                continue
            t = datetime.fromtimestamp(ts, tz=timezone.utc)
            edges.append((src, dst, t))
    return edges


def build_node_index(edges, max_nodes):
    """Keep only the max_nodes most active users (by total in+out degree)."""
    activity = Counter()
    for src, dst, _t in edges:
        activity[src] += 1
        activity[dst] += 1
    top_nodes = [name for name, _ in activity.most_common(max_nodes)]
    return {name: i for i, name in enumerate(top_nodes)}


def make_snapshots(edges, node_to_id, n_snapshots, window_days, start_day_offset):
    """Slice a contiguous n_snapshots * window_days window, starting
    start_day_offset days after the dataset's first timestamp (to skip
    the sparser early ramp-up period), and bin edges into it."""
    filtered = [(s, d, t) for s, d, t in edges if s in node_to_id and d in node_to_id]
    filtered.sort(key=lambda e: e[2])
    if not filtered:
        raise ValueError("No edges left after filtering -- check max_nodes / file path.")

    t_first = filtered[0][2]
    window_start = t_first + timedelta(days=start_day_offset)
    window_seconds = window_days * 24 * 3600
    n = len(node_to_id)

    snapshots = [nx.DiGraph() for _ in range(n_snapshots)]
    for G in snapshots:
        G.add_nodes_from(range(n))

    for src, dst, t in filtered:
        if t < window_start:
            continue
        elapsed = (t - window_start).total_seconds()
        idx = int(elapsed // window_seconds)
        if idx >= n_snapshots:
            break
        snapshots[idx].add_edge(node_to_id[src], node_to_id[dst])

    total_days = n_snapshots * window_days
    print(f"  window: day {start_day_offset} to day {start_day_offset + total_days} "
          f"({total_days} days = {total_days/365.25:.2f} years)")
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_path", help="path to the unzipped sx-*.txt file")
    parser.add_argument("--max_nodes", type=int, default=40000)
    parser.add_argument("--n_snapshots", type=int, default=13)
    parser.add_argument("--window_days", type=int, default=90)
    parser.add_argument("--start_day_offset", type=int, default=545,
                         help="skip this many days from the dataset's start, to land in a "
                              "denser/more mature activity period rather than the sparse "
                              "early ramp-up")
    parser.add_argument("--out", default="sx_snapshots.pkl")
    args = parser.parse_args()

    print(f"Loading edges from {args.txt_path} ...")
    edges = load_edges(args.txt_path)
    print(f"  {len(edges)} raw timestamped edges")

    node_to_id = build_node_index(edges, args.max_nodes)
    print(f"  keeping top {len(node_to_id)} most active users as nodes")

    snapshots = make_snapshots(edges, node_to_id, args.n_snapshots, args.window_days, args.start_day_offset)
    for i, G in enumerate(snapshots):
        print(f"  snapshot {i}: {G.number_of_edges()} edges")

    with open(args.out, "wb") as f:
        pickle.dump(snapshots, f)
    print(f"\nSaved {len(snapshots)} snapshots to {args.out}")

    id_to_name = {i: name for name, i in node_to_id.items()}
    names_out = args.out.replace(".pkl", "_node_names.pkl")
    with open(names_out, "wb") as f:
        pickle.dump(id_to_name, f)
    print(f"Saved node ID -> user ID mapping to {names_out}")

    print("\nTrain on this with:")
    print(f"    python train_world_model.py --real_data {args.out} --node_names {names_out} --celf_snapshots 4")


if __name__ == "__main__":
    main()