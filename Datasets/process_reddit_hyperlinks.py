"""
Convert the SNAP soc-RedditHyperlinks dataset (a REAL 3.25-year dynamic
social network: Jan 2014 - Apr 2017, 55,863 nodes, 858,490 edges) into a
sequence of graph snapshots compatible with dynamic_graph.py / world_model.py.

"""

import argparse
import pickle
from collections import Counter
from datetime import datetime

import networkx as nx


def load_edges(path):
   
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header row
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            src, dst, _post_id, ts = parts[0], parts[1], parts[2], parts[3]
            try:
                t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            edges.append((src, dst, t))
    return edges


def build_node_index(edges, max_nodes):
    """Keep only the max_nodes most active subreddits (by total degree),
    so the graph stays a tractable size for the GCN encoder."""
    activity = Counter()
    for src, dst, _t in edges:
        activity[src] += 1
        activity[dst] += 1
    top_nodes = [name for name, _ in activity.most_common(max_nodes)]
    node_to_id = {name: i for i, name in enumerate(top_nodes)}
    return node_to_id


def make_snapshots(edges, node_to_id, n_snapshots, window_days):
    
    filtered = [(src, dst, t) for src, dst, t in edges if src in node_to_id and dst in node_to_id]
    filtered.sort(key=lambda e: e[2])
    if not filtered:
        raise ValueError("No edges left after filtering -- check max_nodes / file path.")

    t_start = filtered[0][2]
    window = window_days * 24 * 3600  # seconds
    n = len(node_to_id)

    snapshots = [nx.DiGraph() for _ in range(n_snapshots)]
    for G in snapshots:
        G.add_nodes_from(range(n))  # keep node set identical across snapshots

    for src, dst, t in filtered:
        elapsed = (t - t_start).total_seconds()
        snap_idx = int(elapsed // window)
        if snap_idx >= n_snapshots:
            break  # only take the first n_snapshots * window_days of data
        snapshots[snap_idx].add_edge(node_to_id[src], node_to_id[dst])

    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", help="path to soc-redditHyperlinks-*.tsv")
    parser.add_argument("--max_nodes", type=int, default=2000)
    parser.add_argument("--n_snapshots", type=int, default=16)
    parser.add_argument("--window_days", type=int, default=180,
                         help="days per snapshot (180 days x 16 ~= 8 years of coverage window; "
                              "reduce n_snapshots or window_days to match the ~3.25yr real span)")
    parser.add_argument("--out", default="reddit_snapshots.pkl")
    args = parser.parse_args()

    print(f"Loading edges from {args.tsv_path} ...")
    edges = load_edges(args.tsv_path)
    print(f"  {len(edges)} raw timestamped edges")

    node_to_id = build_node_index(edges, args.max_nodes)
    print(f"  keeping top {len(node_to_id)} most active subreddits as nodes")

    snapshots = make_snapshots(edges, node_to_id, args.n_snapshots, args.window_days)
    for i, G in enumerate(snapshots):
        print(f"  snapshot {i}: {G.number_of_edges()} edges")

    with open(args.out, "wb") as f:
        pickle.dump(snapshots, f)
    print(f"\nSaved {len(snapshots)} snapshots to {args.out}")

    id_to_name = {i: name for name, i in node_to_id.items()}
    names_out = args.out.replace(".pkl", "_node_names.pkl")
    with open(names_out, "wb") as f:
        pickle.dump(id_to_name, f)
    print(f"Saved node ID -> subreddit name mapping to {names_out}")

    print("\nLoad them in train_world_model.py with:")
    print(f"    python train_world_model.py --real_data {args.out} --node_names {names_out}")


if __name__ == "__main__":
    main()