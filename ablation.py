"""
ablation.py  —  WWW 2027 Ablation Study
========================================
World Model only. Tests three ablation axes:

  A — Encoder depth   : 1-layer | 2-layer (baseline) | 3-layer
  B — Node features   : degree-only | deg+in+out | full-4 (baseline)
  C — Decoder capacity: dot-product | 1-layer MLP | 2-layer MLP (baseline)

Protocol: same 30-day / 18-5-16 split, 3 seeds, random negatives.

Usage:
  python ablation.py --data_path reddit_30day.pkl    --dataset_name Reddit
  python ablation.py --data_path superuser_30day.pkl --dataset_name SuperUser
  python ablation.py --data_path askubuntu_30day.pkl --dataset_name AskUbuntu
"""

import os, sys, argparse, pickle, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from link_prediction import (
    build_edge_index, sample_negative_edges,
    compute_ap, compute_auc, compute_mrr,
    edge_index_to_sparse_norm_adj,
)

# Import GCNEncoder from worldmodel for A2 baseline
try:
    from worldmodel import GCNEncoder as _WM_GCNEncoder
    _HAS_WM = True
except ImportError:
    _HAS_WM = False

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRAIN = 18
N_VAL   = 5
SEEDS   = [42, 123, 2024]


# ─────────────────────────────────────────────────────────────────────────────
#  Ablation B – Feature extraction variants
# ─────────────────────────────────────────────────────────────────────────────
def get_features(G, num_nodes, feat_dim, mode, device='cpu'):
    """
    mode = 'degree_only'  : x[:,0] = log(1+deg),  rest 0
           'degree_trio'  : x[:,0:3] = log(1+deg/in/out), rest 0
           'full'         : x[:,0:4] = log(1+deg/in/out/avg_nb_deg)  ← baseline
    """
    x = torch.zeros(num_nodes, feat_dim, device=device)
    for n in G.nodes():
        if n >= num_nodes:
            continue
        deg     = G.degree(n)
        in_deg  = G.in_degree(n)  if G.is_directed() else deg
        out_deg = G.out_degree(n) if G.is_directed() else deg
        x[n, 0] = math.log1p(deg)
        if mode in ('degree_trio', 'full'):
            x[n, 1] = math.log1p(in_deg)
            x[n, 2] = math.log1p(out_deg)
        if mode == 'full':
            nbrs = list(G.neighbors(n))
            avg_nb = sum(G.degree(nb) for nb in nbrs) / max(deg, 1)
            x[n, 3] = math.log1p(avg_nb)
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Ablation A – Variable-depth GCN encoder
# ─────────────────────────────────────────────────────────────────────────────
class GCNEncoderDepth(nn.Module):
    """GCN encoder with configurable number of layers (1, 2, or 3)."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        assert n_layers in (1, 2, 3)
        self.n_layers = n_layers
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(n_layers)])

    @staticmethod
    def _prop(h, edge_index, num_nodes):
        if edge_index.size(1) == 0:
            return h
        src, dst = edge_index
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.size(1)), h[src])
        deg = torch.zeros(num_nodes, device=h.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=h.device))
        return h + agg / deg.clamp(min=1).unsqueeze(1)

    def forward(self, x, edge_index, num_nodes):
        h = x
        for i, layer in enumerate(self.layers):
            h = self._prop(h, edge_index, num_nodes)
            h = layer(h)
            if i < self.n_layers - 1:
                h = F.relu(h)
        return h


# ─────────────────────────────────────────────────────────────────────────────
#  Ablation C – Variable-capacity decoder
# ─────────────────────────────────────────────────────────────────────────────
class AblationWorldModel(nn.Module):
    """
    World Model with configurable encoder depth and decoder type.
    encoder_layers : 1 | 2 | 3
    decoder_type   : 'dot' | 'mlp1' | 'mlp2'
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 encoder_layers=2, decoder_type='mlp2'):
        super().__init__()
        self.decoder_type = decoder_type

        # ── Encoder ──────────────────────────────────────────────────────────
        # For A2 (baseline), use worldmodel.py's GCNEncoder if available;
        # otherwise fall back to local variable-depth encoder.
        if encoder_layers == 2 and _HAS_WM:
            self.encoder = _WM_GCNEncoder(in_dim, hidden_dim, out_dim)
            self._use_wm  = True
        else:
            self.encoder = GCNEncoderDepth(in_dim, hidden_dim, out_dim,
                                           n_layers=encoder_layers)
            self._use_wm  = False

        # ── Decoder ──────────────────────────────────────────────────────────
        if decoder_type == 'dot':
            self.decoder = None          # dot-product has no parameters
        elif decoder_type == 'mlp1':
            self.decoder = nn.Linear(out_dim * 2, 1)
        else:  # mlp2 — baseline
            self.decoder = nn.Sequential(
                nn.Linear(out_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )

    def _encode(self, x, edge_index, num_nodes, A_norm=None):
        if self._use_wm and A_norm is not None:
            return self.encoder(A_norm, x)           # worldmodel.py signature
        return self.encoder(x, edge_index, num_nodes)

    def forward(self, x, edge_index, num_nodes, src_nodes, dst_nodes,
                A_norm=None, **kw):
        h = self._encode(x, edge_index, num_nodes, A_norm)
        if self.decoder_type == 'dot':
            return (h[src_nodes] * h[dst_nodes]).sum(-1)
        return self.decoder(
            torch.cat([h[src_nodes], h[dst_nodes]], dim=1)).squeeze(1)

    @torch.no_grad()
    def predict(self, x, edge_index, num_nodes, src_nodes, dst_nodes,
                A_norm=None, **kw):
        return torch.sigmoid(
            self.forward(x, edge_index, num_nodes, src_nodes, dst_nodes,
                         A_norm=A_norm)).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_snaps, num_nodes, epochs, lr,
                feat_dim, feat_mode, seed):
    model.train().to(DEVICE)
    opt  = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    rng  = np.random.default_rng(seed)

    cache = []
    for i in range(len(train_snaps) - 1):
        src_G, tgt_G = train_snaps[i], train_snaps[i+1]
        x      = get_features(src_G, num_nodes, feat_dim, feat_mode, str(DEVICE))
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
                model(x, ei, num_nodes, src, dst, A_norm=A_norm, snap_idx=i), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        if epoch % max(1, epochs//5) == 0:
            print(f"        epoch {epoch:3d}/{epochs}  "
                  f"loss={total/max(nb,1):.4f}")
    model.eval()


def eval_transitions(model, test_pairs, num_nodes, feat_dim,
                     feat_mode, eval_seed):
    model.eval()
    out = []
    for i, (src_G, tgt_G) in enumerate(test_pairs):
        rng  = np.random.default_rng(eval_seed + i)
        pos  = [(u,v) for u,v in tgt_G.edges() if u < num_nodes and v < num_nodes]
        neg  = sample_negative_edges(tgt_G, num_nodes, len(pos), rng)
        if not pos + neg: continue
        lbls = [1]*len(pos) + [0]*len(neg)
        x    = get_features(src_G, num_nodes, feat_dim, feat_mode, str(DEVICE))
        ei   = build_edge_index(src_G, num_nodes, str(DEVICE))
        A_nm = edge_index_to_sparse_norm_adj(ei, num_nodes)
        src  = torch.tensor([e[0] for e in pos+neg], dtype=torch.long, device=DEVICE)
        dst  = torch.tensor([e[1] for e in pos+neg], dtype=torch.long, device=DEVICE)
        sc   = model.predict(x, ei, num_nodes, src, dst, A_norm=A_nm, snap_idx=i)
        out.append((compute_ap(lbls,sc), compute_auc(lbls,sc), compute_mrr(lbls,sc)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Ablation configurations
# ─────────────────────────────────────────────────────────────────────────────
def get_ablation_configs(feat_dim, hidden, out_dim):
    """
    Returns list of (group, label, model_kwargs, feat_mode).
    baseline is A2 / B3 / C3.
    """
    configs = [
        # ── A: Encoder depth ────────────────────────────────────────────────
        ('A', '1-layer GCN',            dict(encoder_layers=1, decoder_type='mlp2'), 'full'),
        ('A', '2-layer GCN (baseline)', dict(encoder_layers=2, decoder_type='mlp2'), 'full'),
        ('A', '3-layer GCN',            dict(encoder_layers=3, decoder_type='mlp2'), 'full'),
        # ── B: Node features ────────────────────────────────────────────────
        ('B', 'degree only',            dict(encoder_layers=2, decoder_type='mlp2'), 'degree_only'),
        ('B', 'deg+in+out',             dict(encoder_layers=2, decoder_type='mlp2'), 'degree_trio'),
        ('B', 'full features (baseline)',dict(encoder_layers=2, decoder_type='mlp2'), 'full'),
        # ── C: Decoder capacity ─────────────────────────────────────────────
        ('C', 'dot-product',            dict(encoder_layers=2, decoder_type='dot'),  'full'),
        ('C', '1-layer MLP',            dict(encoder_layers=2, decoder_type='mlp1'), 'full'),
        ('C', '2-layer MLP (baseline)', dict(encoder_layers=2, decoder_type='mlp2'), 'full'),
    ]
    return configs


# ─────────────────────────────────────────────────────────────────────────────
#  Run one seed for all ablation configs
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation_seed(seed, train_snaps, val_snaps, test_pairs,
                      num_nodes, epochs, hidden, feat_dim, lr,
                      configs):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    tv        = train_snaps + val_snaps
    eval_seed = seed + 1
    out       = {}
    out_dim   = hidden // 2

    for group, label, mkw, feat_mode in configs:
        key = f"{group}:{label}"
        print(f"\n    [{group}] {label}  (feat={feat_mode}) …")
        t0 = time.perf_counter()
        m  = AblationWorldModel(feat_dim, hidden, out_dim, **mkw)
        train_model(m, tv, num_nodes, epochs, lr, feat_dim, feat_mode, seed)
        tr = eval_transitions(m, test_pairs, num_nodes, feat_dim,
                              feat_mode, eval_seed)
        out[key] = {'transitions': tr, 'time': time.perf_counter()-t0,
                    'group': group, 'label': label}
        ap=[t[0] for t in tr]; auc=[t[1] for t in tr]; mrr=[t[2] for t in tr]
        print(f"      AP={np.mean(ap):.4f}±{np.std(ap):.4f}  "
              f"    AUC={np.mean(auc):.4f}±{np.std(auc):.4f}  "
              f"     MRR={np.mean(mrr):.4f}±{np.std(mrr):.4f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate & print
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(seed_runs, configs):
    agg = {}
    for group, label, _, _ in configs:
        key = f"{group}:{label}"
        ap, auc, mrr, t = [], [], [], []
        for run in seed_runs:
            for a, u, r in run[key]['transitions']:
                ap.append(a); auc.append(u); mrr.append(r)
            t.append(run[key]['time'])
        agg[key] = dict(
            group=group, label=label,
            ap_mean=np.mean(ap),   ap_std=np.std(ap),
            auc_mean=np.mean(auc), auc_std=np.std(auc),
            mrr_mean=np.mean(mrr), mrr_std=np.std(mrr),
            time_mean=np.mean(t),  time_std=np.std(t),
        )
    return agg


def print_ablation_table(dataset_name, agg, configs, n_seeds, n_trans):
    LW, CW = 30, 18
    W   = LW + 3*(CW+1) + 14
    bar = '═'*W; thin = '─'*W

    for grp, grp_name in [('A','Encoder Depth'),
                           ('B','Node Features'),
                           ('C','Decoder Capacity')]:
        print(f"\n\n{bar}")
        print(f"  ABLATION {grp}: {grp_name}  —  {dataset_name}")
        print(f"  {n_seeds} seeds × {n_trans} transitions = "
              f"{n_seeds*n_trans} evals/cell")
        print(bar)
        print(f"{'Variant':<{LW}}"
              f"{'Avg.Prec (mean±std)':>{CW}} "
              f"{'AUC-ROC (mean±std)':>{CW}} "
              f"{'MRR (mean±std)':>{CW}}")
        print(thin)
        for group, label, _, _ in configs:
            if group != grp: continue
            key = f"{group}:{label}"
            m   = agg[key]
            is_base = 'baseline' in label
            row = (f"{'▶ ' if is_base else '  '}{label:<{LW-2}}"
                   f"{m['ap_mean']:>7.4f}±{m['ap_std']:.4f}  "
                   f"   {m['auc_mean']:>7.4f}±{m['auc_std']:.4f}  "
                   f"       {m['mrr_mean']:>7.4f}±{m['mrr_std']:.4f}")
            print(row)
        print(bar)
    print("▶ = baseline (full World Model)\n")


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

    with open(args.data_path, 'rb') as f:
        snaps = pickle.load(f)
    if isinstance(snaps, dict):
        snaps = snaps.get('graphs', next(v for v in snaps.values()
                                         if isinstance(v, list)))

    num_nodes   = max((max(G.nodes(), default=-1) for G in snaps), default=-1) + 1
    train_snaps = snaps[:N_TRAIN]
    val_snaps   = snaps[N_TRAIN:N_TRAIN+N_VAL]
    test_snaps  = snaps[N_TRAIN+N_VAL:]
    test_pairs  = [(test_snaps[i], test_snaps[i+1])
                   for i in range(len(test_snaps)-1)]
    n_trans     = len(test_pairs)

    configs = get_ablation_configs(args.feat_dim, args.hidden, args.hidden//2)

    print(f"\n{'═'*60}")
    print(f"  ABLATION STUDY  —  {args.dataset_name}")
    print(f"  Nodes: {num_nodes}  |  Test transitions: {n_trans}")
    print(f"  Seeds: {SEEDS}  |  Configs: {len(configs)}")
    print(f"{'═'*60}")

    seed_runs = []
    for idx, seed in enumerate(SEEDS):
        print(f"\n{'─'*60}")
        print(f"  SEED {seed}  ({idx+1}/{len(SEEDS)})")
        print(f"{'─'*60}")
        seed_runs.append(run_ablation_seed(
            seed, train_snaps, val_snaps, test_pairs,
            num_nodes, args.epochs, args.hidden, args.feat_dim, args.lr,
            configs))

    agg = aggregate(seed_runs, configs)
    print_ablation_table(args.dataset_name, agg, configs, len(SEEDS), n_trans)


if __name__ == '__main__':
    main()
