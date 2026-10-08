"""
rollout.py  —  WWW 2027 Multi-step Rollout Evaluation
======================================================
Tests the World Model's ability to predict multiple steps ahead by
feeding its own predictions back as input (model-based rollout).

Steps evaluated:
  t+1 : predict from real     G_t         (standard link prediction)
  t+2 : predict from predicted G_{t+1}    (1 imagined step)
  t+3 : predict from predicted G_{t+2}    (2 imagined steps)

Comparison: World Model vs GraphMixer (strongest accuracy baseline)
           across all 3 rollout horizons.

Usage:
  python rollout.py --data_path reddit_30day.pkl    --dataset_name Reddit
  python rollout.py --data_path superuser_30day.pkl --dataset_name SuperUser
  python rollout.py --data_path askubuntu_30day.pkl --dataset_name AskUbuntu
"""

import os, sys, argparse, pickle, time, random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.optim import Adam

from link_prediction import (
    GCNLinkPredictor, GraphMixerPredictor,
    build_edge_index, get_node_features, sample_negative_edges,
    compute_ap, compute_auc, compute_mrr, edge_index_to_sparse_norm_adj,
)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRAIN = 18
N_VAL   = 5
SEEDS   = [42, 123, 2024]
HORIZONS = [1, 2, 3]           # predict t+1, t+2, t+3
THRESHOLD = 0.5                 # score threshold to materialise a predicted edge
METHOD_ORDER = ['GraphMixer', 'World Model']


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_snapshots(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        data = data.get('graphs', next(v for v in data.values()
                                       if isinstance(v, list)))
    return data

def get_num_nodes(snaps):
    return max((max(G.nodes(), default=-1) for G in snaps), default=-1) + 1


# ─────────────────────────────────────────────────────────────────────────────
#  Training  (same protocol as train_link_pred.py)
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_snaps, num_nodes, epochs, lr, feat_dim, seed):
    model.train().to(DEVICE)
    opt  = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    rng  = np.random.default_rng(seed)

    cache = []
    for i in range(len(train_snaps) - 1):
        src_G, tgt_G = train_snaps[i], train_snaps[i+1]
        x      = get_node_features(src_G, num_nodes, feat_dim, str(DEVICE))
        ei     = build_edge_index(src_G, num_nodes, str(DEVICE))
        A_norm = edge_index_to_sparse_norm_adj(ei, num_nodes)
        pos    = [(u,v) for u,v in tgt_G.edges()
                  if u < num_nodes and v < num_nodes]
        cache.append((i, x, ei, A_norm, tgt_G, pos))

    for epoch in range(1, epochs+1):
        total, nb = 0.0, 0
        for i, x, ei, A_norm, tgt_G, pos_edges in cache:
            if not pos_edges: continue
            neg = sample_negative_edges(tgt_G, num_nodes, len(pos_edges), rng)
            if not neg: continue
            all_e  = pos_edges + neg
            labels = [1.0]*len(pos_edges) + [0.0]*len(neg)
            src = torch.tensor([e[0] for e in all_e], dtype=torch.long,  device=DEVICE)
            dst = torch.tensor([e[1] for e in all_e], dtype=torch.long,  device=DEVICE)
            lbl = torch.tensor(labels,                dtype=torch.float, device=DEVICE)
            opt.zero_grad()
            loss = crit(
                model(x, ei, num_nodes, src, dst, snap_idx=i, A_norm=A_norm), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        if epoch % max(1, epochs//5) == 0:
            print(f"      epoch {epoch:3d}/{epochs}  "
                  f"loss={total/max(nb,1):.4f}")
    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
#  Materialise a predicted graph from edge scores
# ─────────────────────────────────────────────────────────────────────────────
def scores_to_graph(src_nodes, dst_nodes, scores, num_nodes,
                    threshold=THRESHOLD, directed=True):
    """
    Convert predicted edge probabilities → NetworkX graph.
    Only edges with score ≥ threshold are included.
    This is the 'imagined' next snapshot used for multi-step rollout.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for u, v, s in zip(src_nodes, dst_nodes, scores):
        if s >= threshold:
            G.add_edge(int(u), int(v))
    return G


def predict_full_graph(model, src_G, num_nodes, feat_dim, snap_idx,
                       candidate_sample=5000, rng=None, directed=True):
    """
    Predict all edges for the next snapshot by scoring a large candidate set.

    Because scoring all O(N²) pairs is prohibitive for 100k-node graphs,
    we use a candidate set consisting of:
      - All existing edges in src_G (likely to persist)
      - A random sample of non-edges (may appear next)

    This is the standard 'candidate generation + scoring' approach used
    in large-scale link prediction.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    existing = list(src_G.edges())
    # Sample candidate non-edges
    non_edges = sample_negative_edges(
        src_G, num_nodes, min(candidate_sample, num_nodes), rng)
    candidates = existing + non_edges

    if not candidates:
        return nx.DiGraph() if directed else nx.Graph()

    x   = get_node_features(src_G, num_nodes, feat_dim, str(DEVICE))
    ei  = build_edge_index(src_G, num_nodes, str(DEVICE))
    src = torch.tensor([e[0] for e in candidates], dtype=torch.long, device=DEVICE)
    dst = torch.tensor([e[1] for e in candidates], dtype=torch.long, device=DEVICE)

    scores = model.predict(x, ei, num_nodes, src, dst, snap_idx=snap_idx)
    return scores_to_graph(
        [e[0] for e in candidates],
        [e[1] for e in candidates],
        scores, num_nodes,
        directed=src_G.is_directed()
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-step rollout evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_rollout(model, test_snaps, num_nodes, feat_dim, eval_seed):
    """
    For each anchor snapshot G_t in the test sequence, evaluate prediction
    at horizons h=1,2,3.

    Horizon 1: score edges against real G_{t+1}
    Horizon 2: materialise G_{t+1} from predictions, score against real G_{t+2}
    Horizon 3: materialise G_{t+2} from predictions, score against real G_{t+3}

    Returns: {horizon: [(ap,auc,mrr), ...]}  — one tuple per anchor point.
    """
    model.eval()
    results = {h: [] for h in HORIZONS}
    rng     = np.random.default_rng(eval_seed)

    # Need at least 4 snapshots: t, t+1, t+2, t+3
    max_anchor = len(test_snaps) - max(HORIZONS)

    for anchor in range(max_anchor):
        G_real = [test_snaps[anchor + h] for h in range(max(HORIZONS)+1)]
        # G_real[0]=G_t, G_real[1]=G_{t+1}, G_real[2]=G_{t+2}, G_real[3]=G_{t+3}

        # ── Horizon 1: predict from real G_t ─────────────────────────────────
        G_input = G_real[0]
        G_pred_1 = predict_full_graph(
            model, G_input, num_nodes, feat_dim,
            snap_idx=anchor, rng=rng,
            directed=G_input.is_directed())

        ap, auc, mrr = _score_prediction(
            model, G_input, G_real[1], num_nodes, feat_dim, anchor, rng)
        results[1].append((ap, auc, mrr))

        # ── Horizon 2: predict from predicted G_{t+1} ────────────────────────
        ap, auc, mrr = _score_prediction(
            model, G_pred_1, G_real[2], num_nodes, feat_dim, anchor+1, rng)
        results[2].append((ap, auc, mrr))

        # ── Horizon 3: predict from predicted G_{t+2} ────────────────────────
        G_pred_2 = predict_full_graph(
            model, G_pred_1, num_nodes, feat_dim,
            snap_idx=anchor+1, rng=rng,
            directed=G_pred_1.is_directed())

        ap, auc, mrr = _score_prediction(
            model, G_pred_2, G_real[3], num_nodes, feat_dim, anchor+2, rng)
        results[3].append((ap, auc, mrr))

    return results


def _score_prediction(model, src_G, tgt_G, num_nodes, feat_dim, snap_idx, rng):
    """Score model predictions from src_G against ground truth tgt_G."""
    pos = [(u,v) for u,v in tgt_G.edges() if u < num_nodes and v < num_nodes]
    if not pos:
        return 0.0, 0.5, 0.0
    neg  = sample_negative_edges(tgt_G, num_nodes, len(pos), rng)
    all_e = pos + neg
    lbls  = [1]*len(pos) + [0]*len(neg)
    x   = get_node_features(src_G, num_nodes, feat_dim, str(DEVICE))
    ei  = build_edge_index(src_G, num_nodes, str(DEVICE))
    src = torch.tensor([e[0] for e in all_e], dtype=torch.long, device=DEVICE)
    dst = torch.tensor([e[1] for e in all_e], dtype=torch.long, device=DEVICE)
    sc  = model.predict(x, ei, num_nodes, src, dst, snap_idx=snap_idx)
    return compute_ap(lbls,sc), compute_auc(lbls,sc), compute_mrr(lbls,sc)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-seed run
# ─────────────────────────────────────────────────────────────────────────────
def run_one_seed(seed, train_snaps, val_snaps, test_snaps,
                 num_nodes, epochs, hidden, feat_dim, lr):
    set_seed(seed)
    tv        = train_snaps + val_snaps
    eval_seed = seed + 1
    out       = {}

    # ── GraphMixer ───────────────────────────────────────────────────────────
    print("\n    GraphMixer …")
    t0 = time.perf_counter()
    m  = GraphMixerPredictor(feat_dim, hidden, hidden//2)
    train_model(m, tv, num_nodes, epochs, lr, feat_dim, seed)
    r  = evaluate_rollout(m, test_snaps, num_nodes, feat_dim, eval_seed)
    out['GraphMixer'] = {'rollout': r, 'time': time.perf_counter()-t0}
    _show_rollout(r)

    # ── World Model ──────────────────────────────────────────────────────────
    print("\n    World Model …")
    t0 = time.perf_counter()
    m  = GCNLinkPredictor(feat_dim, hidden, hidden//2)
    train_model(m, tv, num_nodes, epochs, lr, feat_dim, seed)
    r  = evaluate_rollout(m, test_snaps, num_nodes, feat_dim, eval_seed)
    out['World Model'] = {'rollout': r, 'time': time.perf_counter()-t0}
    _show_rollout(r)

    return out


def _show_rollout(rollout_results):
    for h in HORIZONS:
        tr = rollout_results[h]
        if not tr: continue
        ap=[t[0] for t in tr]; auc=[t[1] for t in tr]; mrr=[t[2] for t in tr]
        print(f"      t+{h}: AP={np.mean(ap):.4f}  "
              f"AUC={np.mean(auc):.4f}  MRR={np.mean(mrr):.4f}  "
              f"(n={len(tr)})")


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate & print
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(seed_runs):
    agg = {}
    for method in METHOD_ORDER:
        agg[method] = {}
        for h in HORIZONS:
            ap, auc, mrr = [], [], []
            for run in seed_runs:
                for a, u, r in run[method]['rollout'][h]:
                    ap.append(a); auc.append(u); mrr.append(r)
            agg[method][h] = dict(
                ap_mean=np.mean(ap),   ap_std=np.std(ap),
                auc_mean=np.mean(auc), auc_std=np.std(auc),
                mrr_mean=np.mean(mrr), mrr_std=np.std(mrr),
                n=len(ap)
            )
        times = [run[method]['time'] for run in seed_runs]
        agg[method]['time_mean'] = np.mean(times)
        agg[method]['time_std']  = np.std(times)
    return agg


def print_rollout_table(dataset_name, agg, n_seeds):
    HW, MW, CW = 8, 22, 16
    W   = HW + MW + 3*(CW+1) + 4
    bar = '═'*W; thin = '─'*W

    print(f"\n\n{bar}")
    print(f"  MULTI-STEP ROLLOUT  —  {dataset_name}")
    print(f"  {n_seeds} seeds  |  Horizon: t+1 (real input) / "
          f"t+2 / t+3 (imagined input)")
    print(bar)
    print(f"{'Horizon':<{HW}}{'Method':<{MW}}"
          f"{'Avg.Prec (mean±std)':>{CW}} "
          f"  {'AUC-ROC (mean±std)':>{CW}} "
          f"  {'MRR (mean±std)':>{CW}}")
    print(thin)

    for h in HORIZONS:
        label = f"t+{h}" + (" ← real" if h==1 else " ← imagined")
        for i, method in enumerate(METHOD_ORDER):
            m   = agg[method][h]
            mrk = '◄' if method == 'World Model' else ' '
            row = (f"{label if i==0 else '':<{HW}}"
                   f"{ method:<{MW}}"
                   f"{m['ap_mean']:>7.4f}±{m['ap_std']:.4f}  "
                   f"    {m['auc_mean']:>7.4f}±{m['auc_std']:.4f}  "
                   f"   {m['mrr_mean']:>7.4f}±{m['mrr_std']:.4f} {mrk}")
            print(row)
        if h < max(HORIZONS):
            print(thin)

    print(bar)
    # Degradation summary
    print("\n  Degradation vs t+1 (AP drop per additional step):")
    for method in METHOD_ORDER:
        ap1 = agg[method][1]['ap_mean']
        ap2 = agg[method][2]['ap_mean']
        ap3 = agg[method][3]['ap_mean']
        print(f"    {method:<22} t+2: {ap2-ap1:+.4f}  t+3: {ap3-ap1:+.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_path', '--real_data', dest='data_path', required=True)
    p.add_argument('--dataset_name', default='Dataset')
    p.add_argument('--epochs',   type=int,   default=100)
    p.add_argument('--hidden',   type=int,   default=64)
    p.add_argument('--feat_dim', type=int,   default=16)
    p.add_argument('--lr',       type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.data_path):
        print(f"[ERROR] {args.data_path} not found"); sys.exit(1)

    snaps     = load_snapshots(args.data_path)
    num_nodes = get_num_nodes(snaps)

    train_snaps = snaps[:N_TRAIN]
    val_snaps   = snaps[N_TRAIN:N_TRAIN+N_VAL]
    test_snaps  = snaps[N_TRAIN+N_VAL:]    # 16 snapshots → 13 anchor points for t+3

    n_anchors = len(test_snaps) - max(HORIZONS)

    print(f"\n{'═'*60}")
    print(f"  MULTI-STEP ROLLOUT  —  {args.dataset_name}")
    print(f"  Nodes: {num_nodes}  |  Test snapshots: {len(test_snaps)}")
    print(f"  Anchor points: {n_anchors}  "
          f"(each gives t+1, t+2, t+3 evaluations)")
    print(f"  Seeds: {SEEDS}  |  Methods: {METHOD_ORDER}")
    print(f"{'═'*60}")

    if n_anchors < 1:
        print("[ERROR] Not enough test snapshots for 3-step rollout. "
              "Need ≥4 test snapshots.")
        sys.exit(1)

    seed_runs = []
    for idx, seed in enumerate(SEEDS):
        print(f"\n{'─'*60}")
        print(f"  SEED {seed}  ({idx+1}/{len(SEEDS)})")
        print(f"{'─'*60}")
        seed_runs.append(run_one_seed(
            seed, train_snaps, val_snaps, test_snaps,
            num_nodes, args.epochs, args.hidden, args.feat_dim, args.lr))

    print_rollout_table(args.dataset_name, aggregate(seed_runs), len(SEEDS))


if __name__ == '__main__':
    main()
