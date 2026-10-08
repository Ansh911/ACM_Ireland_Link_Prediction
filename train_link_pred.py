"""
train_link_pred.py  —  WWW 2027 extension
==========================================
30-day snapshots · 15 test transitions · 3 seeds
Negative sampling strategies: Random | Historical | Inductive
Reports mean±std per strategy separately.

Usage:
  python train_link_pred.py --data_path reddit_30day.pkl    --dataset_name Reddit
  python train_link_pred.py --data_path superuser_30day.pkl --dataset_name SuperUser
  python train_link_pred.py --data_path askubuntu_30day.pkl --dataset_name AskUbuntu
"""

import os, sys, argparse, pickle, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from link_prediction import (
    GCNLinkPredictor, DyRepPredictor, GraphMixerPredictor,
    DyGFormerPredictor, TAMIPredictor,
    build_edge_index, get_node_features, sample_negative_edges,
    sample_historical_negatives, sample_inductive_negatives,
    compute_ap, compute_auc, compute_mrr, edge_index_to_sparse_norm_adj,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_TRAIN = 18
N_VAL   = 5

SEEDS        = [42, 123, 2024]
NEG_STRATEGIES = ['random', 'historical', 'inductive']

METHOD_ORDER = ['DyRep', 'GraphMixer', 'DyGFormer', 'TAMI', 'World Model']
METHOD_REF   = {
    'DyRep'      : "(ICLR '19)",
    'GraphMixer' : "(ICLR '23)",
    'DyGFormer'  : "(NeurIPS '23)",
    'TAMI'       : "(NeurIPS '25)",
    'World Model': "(Ours)",
}


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_snapshots(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        data = data.get('graphs', next(v for v in data.values()
                                       if isinstance(v, list)))
    return data

def get_num_nodes(snaps):
    return max((max(G.nodes(), default=-1) for G in snaps), default=-1) + 1

def build_history_edges(train_snaps) -> set:
    """All edges seen across training snapshots."""
    h = set()
    for G in train_snaps:
        for u, v in G.edges():
            h.add((u, v)); h.add((v, u))
    return h

def build_train_nodes(train_snaps) -> set:
    """All nodes seen across training snapshots."""
    nodes = set()
    for G in train_snaps:
        nodes.update(G.nodes())
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
#  Training  (always random negatives — strategy only affects evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def train_neural_model(model, train_snaps, num_nodes, epochs, lr, feat_dim, seed):
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
            neg_edges = sample_negative_edges(
                tgt_G, num_nodes, len(pos_edges), rng)
            if not neg_edges: continue
            all_e  = pos_edges + neg_edges
            labels = [1.0]*len(pos_edges) + [0.0]*len(neg_edges)
            src = torch.tensor([e[0] for e in all_e],
                               dtype=torch.long,  device=DEVICE)
            dst = torch.tensor([e[1] for e in all_e],
                               dtype=torch.long,  device=DEVICE)
            lbl = torch.tensor(labels, dtype=torch.float, device=DEVICE)
            opt.zero_grad()
            loss = crit(
                model(x, ei, num_nodes, src, dst,
                      snap_idx=i, A_norm=A_norm), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        if epoch % max(1, epochs//5) == 0:
            print(f"      epoch {epoch:3d}/{epochs}  "
                  f"loss={total/max(nb,1):.4f}")
    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation  —  one strategy at a time
# ─────────────────────────────────────────────────────────────────────────────
def eval_per_transition(model, test_pairs, num_nodes, feat_dim,
                        eval_seed, strategy,
                        history_edges=None, train_nodes=None):
    """
    Returns list of (AP, AUC, MRR) — one per test transition.
    strategy: 'random' | 'historical' | 'inductive'
    """
    model.eval()
    out = []
    for i, (src_G, tgt_G) in enumerate(test_pairs):
        rng = np.random.default_rng(eval_seed + i)
        pos = [(u,v) for u,v in tgt_G.edges()
               if u < num_nodes and v < num_nodes]
        if not pos:
            continue

        if strategy == 'random':
            neg = sample_negative_edges(tgt_G, num_nodes, len(pos), rng)
        elif strategy == 'historical':
            neg = sample_historical_negatives(
                tgt_G, history_edges, num_nodes, len(pos), rng)
        else:  # inductive
            neg = sample_inductive_negatives(
                tgt_G, train_nodes, num_nodes, len(pos), rng)

        all_e = pos + neg
        lbls  = [1]*len(pos) + [0]*len(neg)
        x   = get_node_features(src_G, num_nodes, feat_dim, str(DEVICE))
        ei  = build_edge_index(src_G, num_nodes, str(DEVICE))
        src = torch.tensor([e[0] for e in all_e],
                           dtype=torch.long, device=DEVICE)
        dst = torch.tensor([e[1] for e in all_e],
                           dtype=torch.long, device=DEVICE)
        sc  = model.predict(x, ei, num_nodes, src, dst, snap_idx=i)
        out.append((compute_ap(lbls,sc), compute_auc(lbls,sc),
                    compute_mrr(lbls,sc)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Single-seed run
# ─────────────────────────────────────────────────────────────────────────────
def run_one_seed(seed, train_snaps, val_snaps, test_pairs,
                 num_nodes, epochs, hidden, feat_dim, lr,
                 history_edges, train_nodes):
    set_seed(seed)
    tv        = train_snaps + val_snaps
    eval_seed = seed + 1

    def _eval_all(model):
        """Evaluate model under all 3 strategies."""
        return {
            s: eval_per_transition(
                model, test_pairs, num_nodes, feat_dim,
                eval_seed, s, history_edges, train_nodes)
            for s in NEG_STRATEGIES
        }

    def _show(strat_results):
        for s, tr in strat_results.items():
            ap=[t[0] for t in tr]; auc=[t[1] for t in tr]; mrr=[t[2] for t in tr]
            print(f"        [{s:10s}] "
                  f"AP={np.mean(ap):.4f}±{np.std(ap):.4f}  "
                  f"AUC={np.mean(auc):.4f}±{np.std(auc):.4f}  "
                  f"MRR={np.mean(mrr):.4f}±{np.std(mrr):.4f}")

    out = {}

    # ── DyRep ────────────────────────────────────────────────────────────────
    print("\n    [1/5]  DyRep …")
    t0 = time.perf_counter()
    m  = DyRepPredictor(feat_dim, hidden, hidden//2)
    train_neural_model(m, tv, num_nodes, epochs, lr, feat_dim, seed)
    sr = _eval_all(m)
    out['DyRep'] = {'strategies': sr, 'time': time.perf_counter()-t0}
    _show(sr)

    # ── GraphMixer ───────────────────────────────────────────────────────────
    print("\n    [2/5]  GraphMixer …")
    t0 = time.perf_counter()
    m  = GraphMixerPredictor(feat_dim, hidden, hidden//2)
    train_neural_model(m, tv, num_nodes, epochs, lr, feat_dim, seed)
    sr = _eval_all(m)
    out['GraphMixer'] = {'strategies': sr, 'time': time.perf_counter()-t0}
    _show(sr)

    # ── DyGFormer ────────────────────────────────────────────────────────────
    print("\n    [3/5]  DyGFormer …")
    t0 = time.perf_counter()
    m  = DyGFormerPredictor(feat_dim, hidden, hidden//2)
    train_neural_model(m, tv, num_nodes, epochs, lr, feat_dim, seed)
    sr = _eval_all(m)
    out['DyGFormer'] = {'strategies': sr, 'time': time.perf_counter()-t0}
    _show(sr)

    # ── TAMI ─────────────────────────────────────────────────────────────────
    print("\n    [4/5]  TAMI …")
    t0 = time.perf_counter()
    m  = TAMIPredictor(feat_dim, hidden, hidden//2)
    m.fit_lha_memory(train_snaps)
    train_neural_model(m, tv, num_nodes, epochs, lr, feat_dim, seed)
    sr = _eval_all(m)
    out['TAMI'] = {'strategies': sr, 'time': time.perf_counter()-t0}
    _show(sr)

    # ── World Model ──────────────────────────────────────────────────────────
    print("\n    [5/5]  World Model …")
    t0 = time.perf_counter()
    m  = GCNLinkPredictor(feat_dim, hidden, hidden//2)
    train_neural_model(m, tv, num_nodes, epochs, lr, feat_dim, seed)
    sr = _eval_all(m)
    out['World Model'] = {'strategies': sr, 'time': time.perf_counter()-t0}
    _show(sr)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate across seeds
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(seed_runs):
    """
    Returns: method -> strategy -> {ap_mean, ap_std, auc_mean, ...}
             method -> time_{mean,std}
    """
    agg = {}
    for method in METHOD_ORDER:
        agg[method] = {}
        for s in NEG_STRATEGIES:
            ap, auc, mrr = [], [], []
            for run in seed_runs:
                for a, u, r in run[method]['strategies'][s]:
                    ap.append(a); auc.append(u); mrr.append(r)
            agg[method][s] = dict(
                ap_mean=np.mean(ap),   ap_std=np.std(ap),
                auc_mean=np.mean(auc), auc_std=np.std(auc),
                mrr_mean=np.mean(mrr), mrr_std=np.std(mrr),
            )
        times = [run[method]['time'] for run in seed_runs]
        agg[method]['time_mean'] = np.mean(times)
        agg[method]['time_std']  = np.std(times)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
#  Results tables  —  one per negative sampling strategy
# ─────────────────────────────────────────────────────────────────────────────
def print_tables(dataset_name, agg, n_seeds, n_trans):
    MW, CW, TCW = 28, 18, 16
    W   = MW + 3*(CW+1) + TCW + 4
    bar = '═'*W; thin = '─'*W

    for strategy in NEG_STRATEGIES:
        print(f"\n\n{bar}")
        print(f"  {dataset_name}  |  Negatives: {strategy.upper()}")
        print(f"  {n_seeds} seeds × {n_trans} transitions = "
              f"{n_seeds*n_trans} evals/cell")
        print(bar)
        print(f"{'Method':<{MW}}"
              f"{'Avg.Prec (mean±std)':>{CW}} "
              f"{'AUC-ROC (mean±std)':>{CW}} "
              f"{'MRR (mean±std)':>{CW}} "
              f"{'Time (s)':>{TCW}}")
        print(thin)
        for method in METHOD_ORDER:
            ref = METHOD_REF.get(method, '')
            m   = agg[method][strategy]
            t   = agg[method]
            row = (f"{method+' '+ref:<{MW}}"
                   f"{m['ap_mean']:>7.4f}±{m['ap_std']:.4f}  "
                   f"    {m['auc_mean']:>7.4f}±{m['auc_std']:.4f}  "
                   f"     {m['mrr_mean']:>7.4f}±{m['mrr_std']:.4f}  "
                   f"     {t['time_mean']:>6.1f}±{t['time_std']:.1f}s")
            print(row + ('  ◄' if method == 'World Model' else ''))
        print(bar)
    print(f"\nTime = training+eval wall-clock, mean±std across {n_seeds} seeds\n")


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


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if not os.path.exists(args.data_path):
        print(f"[ERROR] {args.data_path} not found"); sys.exit(1)

    print(f"\n{'═'*62}")
    print(f"  Dataset : {args.dataset_name}  |  Device: {DEVICE}")
    print(f"  Epochs  : {args.epochs}  Hidden: {args.hidden}  "
          f"FeatDim: {args.feat_dim}  LR: {args.lr}")
    print(f"  Seeds   : {SEEDS}")
    print(f"  Neg strategies: {NEG_STRATEGIES}")
    print(f"{'═'*62}")

    snaps     = load_snapshots(args.data_path)
    num_nodes = get_num_nodes(snaps)

    train_snaps = snaps[:N_TRAIN]
    val_snaps   = snaps[N_TRAIN:N_TRAIN+N_VAL]
    test_snaps  = snaps[N_TRAIN+N_VAL:]
    test_pairs  = [(test_snaps[i], test_snaps[i+1])
                   for i in range(len(test_snaps)-1)]
    n_trans     = len(test_pairs)

    # Pre-build history and train-node sets (used across all seeds)
    history_edges = build_history_edges(train_snaps)
    train_nodes   = build_train_nodes(train_snaps)

    print(f"\n  Snapshots: {len(snaps)}  |  Nodes: {num_nodes}")
    print(f"  Train: {len(train_snaps)}  Val: {len(val_snaps)}  "
          f"Test: {len(test_snaps)}  →  {n_trans} transitions")
    print(f"  History edges: {len(history_edges)//2}  |  "
          f"Train nodes: {len(train_nodes)}")

    seed_runs = []
    for idx, seed in enumerate(SEEDS):
        print(f"\n{'─'*62}")
        print(f"  SEED {seed}  ({idx+1}/{len(SEEDS)})")
        print(f"{'─'*62}")
        seed_runs.append(run_one_seed(
            seed, train_snaps, val_snaps, test_pairs,
            num_nodes, args.epochs, args.hidden, args.feat_dim, args.lr,
            history_edges, train_nodes))

    print_tables(args.dataset_name, aggregate(seed_runs),
                 len(SEEDS), n_trans)


if __name__ == '__main__':
    main()