"""
link_prediction.py
==================
Link Prediction for Temporal Dynamic Networks.

World Model (GCN) compared against 3 published baselines:
  1. EdgeBank    – memory-based, no training   (Poursafaei et al., NeurIPS 2022)
  2. DyRep       – temporal GNN + GRU states   (Trivedi et al.,    ICLR 2019, simplified)
  3. GraphMixer  – MLP-based temporal model    (Cong et al.,       ICLR 2023, simplified)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# ─────────────────────────────────────────────────────────────────────────────
#  GCNEncoder  –  import from worldmodel.py; fall back to local definition
# ─────────────────────────────────────────────────────────────────────────────


try:
    from worldmodel import GCNEncoder
    print("[link_prediction] Loaded GCNEncoder from worldmodel.py")
except ImportError:
    print("[link_prediction] worldmodel.py not found – using local GCNEncoder fallback")

    class GCNEncoder(nn.Module):
        """2-layer GCN encoder (mirrors the worldmodel.py interface)."""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, out_dim)

        @staticmethod
        def _prop(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
            if edge_index.size(1) == 0:
                return x
            src, dst = edge_index
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
            return x + agg / deg.clamp(min=1).unsqueeze(1)

        def forward(self, x, edge_index, num_nodes):
            h = F.relu(self.lin1(self._prop(x, edge_index, num_nodes)))
            return self.lin2(self._prop(h, edge_index, num_nodes))

def edge_index_to_sparse_norm_adj(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    device = edge_index.device
    self_loops = torch.arange(n, device=device)
    if edge_index.size(1) > 0:
        src = torch.cat([edge_index[0], self_loops])
        dst = torch.cat([edge_index[1], self_loops])
    else:
        src = self_loops
        dst = self_loops
    deg = torch.zeros(n, device=device)
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    d_inv_sqrt = deg.clamp(min=1).pow(-0.5)
    weights = d_inv_sqrt[src] * d_inv_sqrt[dst]
    indices = torch.stack([src, dst])
    return torch.sparse_coo_tensor(indices, weights, size=(n, n)).coalesce()
        


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utility: learnable cosine time encoder
# ─────────────────────────────────────────────────────────────────────────────
class TimeEncoder(nn.Module):
    """
    Learnable cosine time encoding (used by both DyRep and GraphMixer).
    Maps a scalar time value to a d-dimensional feature vector via:
        φ(t) = cos(W·t + b)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, dim)
        nn.init.normal_(self.linear.weight, 0, dim ** -0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t : (N,) float  →  (N, dim)"""
        return torch.cos(self.linear(t.float().unsqueeze(-1)))


# ─────────────────────────────────────────────────────────────────────────────
#  World Model Link Predictor  (unchanged architecture)
# ─────────────────────────────────────────────────────────────────────────────
class GCNLinkPredictor(nn.Module):
    """
    Proposed method: GCN encoder (reuses GCNEncoder from worldmodel.py)
    followed by a 2-layer MLP decoder that scores candidate edges.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, num_nodes, src_nodes, dst_nodes, A_norm=None, **kw):
        if A_norm is None:
            A_norm = edge_index_to_sparse_norm_adj(edge_index, num_nodes)
        h = self.encoder(A_norm, x)
        return self.decoder(
            torch.cat([h[src_nodes], h[dst_nodes]], dim=1)
        ).squeeze(1)

    @torch.no_grad()
    def predict(self, x, edge_index, num_nodes, src_nodes, dst_nodes, **kw):
        return torch.sigmoid(
            self.forward(x, edge_index, num_nodes, src_nodes, dst_nodes)
        ).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 1 – DyRep  (simplified)
# ─────────────────────────────────────────────────────────────────────────────
class DyRepPredictor(nn.Module):
    """
    Simplified DyRep  (Trivedi et al., ICLR 2019).

    DyRep's core idea: node representations evolve continuously over time
    via a GRU-based update triggered by observed interactions.  Each event
    (u, v, t) causes both endpoints to update their hidden state by fusing
    their current structural neighbourhood embedding with a temporal signal.

    Simplifications made here (appropriate for snapshot graphs):
      • Continuous event times are replaced by snapshot index (a discrete
        proxy for time), encoded via a learnable cosine TimeEncoder.
      • The per-snapshot GCN provides structural embeddings; a GRU-style
        gated update then fuses them with the temporal encoding.
      • Link scores are produced by an MLP intensity decoder over pairs of
        updated node representations.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # Structural message passing
        self.gcn_lin1 = nn.Linear(in_dim,      hidden_dim)
        self.gcn_lin2 = nn.Linear(hidden_dim,  hidden_dim)

        # Temporal encoder
        self.time_enc = TimeEncoder(hidden_dim)

        # GRU-style gated temporal state update
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reset_gate  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.candidate   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm        = nn.LayerNorm(hidden_dim)

        # Intensity / link decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def _gcn(self, x: torch.Tensor, edge_index: torch.Tensor,
             num_nodes: int) -> torch.Tensor:
        """Two-layer GCN with skip connection."""
        h = F.relu(self.gcn_lin1(x))
        if edge_index.size(1) > 0:
            src, dst = edge_index
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.size(1)), h[src])
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
            h = F.relu(h + agg / deg.clamp(min=1).unsqueeze(1))
        return F.relu(self.gcn_lin2(h))

    # ------------------------------------------------------------------
    def forward(self, x, edge_index, num_nodes, src_nodes, dst_nodes,
                snap_idx: int = 0, A_norm=None, **kw):
        if A_norm is None:
            A_norm = edge_index_to_sparse_norm_adj(edge_index, num_nodes)
        # Structural embedding
        h = self._gcn(x, edge_index, num_nodes)

        # Temporal encoding (snapshot index as proxy for continuous time)
        t     = torch.full((num_nodes,), float(snap_idx), device=x.device)
        t_enc = self.time_enc(t)

        # GRU-style gated update: fuse structural h with temporal t_enc
        cat_ht  = torch.cat([h, t_enc], dim=1)
        z       = torch.sigmoid(self.update_gate(cat_ht))          # update gate
        r       = torch.sigmoid(self.reset_gate(cat_ht))           # reset gate
        h_cand  = torch.tanh(self.candidate(
                      torch.cat([r * h, t_enc], dim=1)))           # candidate
        node_emb = self.norm((1 - z) * h + z * h_cand)            # updated state

        edge_feat = torch.cat([node_emb[src_nodes], node_emb[dst_nodes]], dim=1)
        return self.decoder(edge_feat).squeeze(1)

    @torch.no_grad()
    def predict(self, x, edge_index, num_nodes, src_nodes, dst_nodes, A_norm=None, **kw):
        return torch.sigmoid(
            self.forward(x, edge_index, num_nodes, src_nodes, dst_nodes, A_norm=None, **kw)
        ).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 3 – GraphMixer  (simplified)
# ─────────────────────────────────────────────────────────────────────────────
class GraphMixerPredictor(nn.Module):
    """
    Simplified GraphMixer  (Cong et al., ICLR 2023).

    GraphMixer's core idea: replace graph convolution entirely with
    MLP-based 'token mixing' and 'channel mixing' (borrowing from
    MLP-Mixer in vision), augmented by a sinusoidal time encoding.
    This means *no adjacency-matrix operations* at all – only MLPs.

    Architecture implemented here:
      1. TimeEncoder:   snapshot index → d-dim cosine features
      2. Token mixer:   MLP over (node_feat ∥ time_feat) with GELU + LayerNorm
      3. Channel mixer: residual MLP over the hidden dimension
      4. Link MLP:      concatenated pair embeddings → link score
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 time_dim: int = 32):
        super().__init__()
        self.time_enc = TimeEncoder(time_dim)

        # Token mixer (node features + time encoding → hidden)
        self.token_mix = nn.Sequential(
            nn.Linear(in_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Channel mixer (residual MLP over hidden channels)
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim,     hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Link-level MLP
        self.link_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,     hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, num_nodes, src_nodes, dst_nodes,
                snap_idx: int = 0, **kw):
        # Time encoding
        t     = torch.full((num_nodes,), float(snap_idx), device=x.device)
        t_enc = self.time_enc(t)

        # Token mixing:  (x ∥ t_enc) → hidden
        h = self.norm1(self.token_mix(torch.cat([x, t_enc], dim=1)))

        # Channel mixing with residual
        h = self.norm2(h + self.channel_mix(h))

        return self.link_mlp(
            torch.cat([h[src_nodes], h[dst_nodes]], dim=1)
        ).squeeze(1)

    @torch.no_grad()
    def predict(self, x, edge_index, num_nodes, src_nodes, dst_nodes, **kw):
        return torch.sigmoid(
            self.forward(x, edge_index, num_nodes, src_nodes, dst_nodes, **kw)
        ).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_ap(y_true, y_scores) -> float:
    """Average Precision (area under precision-recall curve)."""
    y_true   = np.asarray(y_true,   dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_scores))


def compute_auc(y_true, y_scores) -> float:
    """Area under the ROC curve."""
    y_true   = np.asarray(y_true,   dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_scores))


def compute_mrr(y_true, y_scores, num_neg_samples: int = 99,
                seed: int = 42) -> float:
    """
    Mean Reciprocal Rank (MRR) for link prediction.

    For each positive edge, sample `num_neg_samples` negative edges at
    random and compute the rank of the positive edge among all of them.
    MRR = mean(1 / rank).  Higher is better; perfect score = 1.0.
    """
    y_true   = np.asarray(y_true,   dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    pos_idx  = np.where(y_true == 1)[0]
    neg_idx  = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    rrs = []
    for pi in pos_idx:
        neg_samp   = rng.choice(neg_idx,
                                size=min(num_neg_samples, len(neg_idx)),
                                replace=False)
        all_scores = np.concatenate([[y_scores[pi]], y_scores[neg_samp]])
        rank       = int(np.sum(all_scores > y_scores[pi])) + 1   # 1 = best
        rrs.append(1.0 / rank)
    return float(np.mean(rrs))


# ─────────────────────────────────────────────────────────────────────────────
#  Data utilities
# ─────────────────────────────────────────────────────────────────────────────
def build_edge_index(G, num_nodes: int,
                     device: str = 'cpu') -> torch.Tensor:
    """Convert NetworkX graph → bidirectional edge_index tensor."""
    edges = [(u, v) for u, v in G.edges()
             if u < num_nodes and v < num_nodes]
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor(
        [src + dst, dst + src], dtype=torch.long, device=device
    )


def get_node_features(G, num_nodes: int, feat_dim: int = 16,
                      device: str = 'cpu') -> torch.Tensor:
    """
    Simple log-degree node features.
    Feature dim 0 = log(1 + degree); remaining dims are zero-padded.
    """
    x = torch.zeros(num_nodes, feat_dim, device=device)
    for n in G.nodes():
        if n < num_nodes:
            deg    = G.degree(n)
            in_deg = G.in_degree(n)  if G.is_directed() else deg
            out_deg= G.out_degree(n) if G.is_directed() else deg
            nbrs   = list(G.neighbors(n))
            avg_nb_deg = sum(G.degree(nb) for nb in nbrs) / max(deg, 1)

            x[n, 0] = math.log1p(deg)
            x[n, 1] = math.log1p(in_deg)
            x[n, 2] = math.log1p(out_deg)
            x[n, 3] = math.log1p(avg_nb_deg)   # avg neighbour degree
    return x


def sample_negative_edges(G, num_nodes: int, n_neg: int,
                          rng=None) -> list:
    """
    Uniform random sampling of non-existing edges (hard negatives avoided
    by checking against the current snapshot's edge set).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    existing = set()
    for u, v in G.edges():
        existing.add((u, v))
        existing.add((v, u))
    neg, tries = [], 0
    while len(neg) < n_neg and tries < n_neg * 20:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u != v and (u, v) not in existing:
            neg.append((u, v))
            existing.add((u, v))
        tries += 1
    return neg

def sample_historical_negatives(tgt_G, history_edges: set,
                                num_nodes: int, n_neg: int,
                                rng=None) -> list:
    """
    Historical negatives  (Poursafaei et al., NeurIPS 2022).
    Edges that appeared in at least one past training snapshot but are
    NOT present in the current target snapshot.  Harder than random
    because the model has already encoded these edges structurally.
    Falls back to random if not enough historical candidates exist.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    target = set()
    for u, v in tgt_G.edges():
        target.add((u, v)); target.add((v, u))
    candidates = [(u, v) for (u, v) in history_edges
                  if (u, v) not in target
                  and u < num_nodes and v < num_nodes]
    if len(candidates) >= n_neg:
        idx = rng.choice(len(candidates), size=n_neg, replace=False)
        return [candidates[i] for i in idx]
    # not enough — pad with random
    neg = list(candidates)
    neg += sample_negative_edges(tgt_G, num_nodes, n_neg - len(neg), rng)
    return neg


def sample_inductive_negatives(tgt_G, train_nodes: set,
                                num_nodes: int, n_neg: int,
                                rng=None) -> list:
    """
    Inductive negatives  (Poursafaei et al., NeurIPS 2022).
    Edges where at least one endpoint is a node NOT seen during training.
    Tests generalisation to unseen nodes.
    Falls back to random if the target snapshot has no new nodes.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    new_nodes = [n for n in tgt_G.nodes()
                 if n not in train_nodes and n < num_nodes]
    if not new_nodes:
        return sample_negative_edges(tgt_G, num_nodes, n_neg, rng)
    new_arr = np.array(new_nodes)
    target  = set()
    for u, v in tgt_G.edges():
        target.add((u, v)); target.add((v, u))
    existing = set(target)
    neg, tries = [], 0
    while len(neg) < n_neg and tries < n_neg * 20:
        if rng.random() < 0.5:
            u = int(rng.choice(new_arr))
            v = int(rng.integers(0, num_nodes))
        else:
            u = int(rng.integers(0, num_nodes))
            v = int(rng.choice(new_arr))
        if u != v and u < num_nodes and v < num_nodes \
                and (u, v) not in existing:
            neg.append((u, v))
            existing.add((u, v))
        tries += 1
    if len(neg) < n_neg:
        neg += sample_negative_edges(tgt_G, num_nodes, n_neg - len(neg), rng)
    return neg[:n_neg]


# ─────────────────────────────────────────────────────────────────────────────
#  LTE: Log Time Encoder  (TAMI, NeurIPS 2025)
# ─────────────────────────────────────────────────────────────────────────────
class LogTimeEncoder(nn.Module):
    """
    LTE: Log Time Encoding from TAMI (Yu et al., NeurIPS 2025).

    Standard time encoding:  z(t) = cos(W · Δt)
    LTE:                     z(t) = cos(W · ln(1+Δt))

    The logarithmic transformation reduces skewness in temporal differences
    (proven to reduce skewness of a Pareto-distributed Δt to exactly 2),
    making the learnable frequency parameters ω easier to optimise —
    especially for infrequently interacting node pairs.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, dim)
        nn.init.normal_(self.linear.weight, 0, dim ** -0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (N,) float → (N, dim)"""
        t_log = torch.log1p(t.float())          # ln(1+t) — the TAMI key step
        return torch.cos(self.linear(t_log.unsqueeze(-1)))


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 4 – DyGFormer  (simplified)
# ─────────────────────────────────────────────────────────────────────────────
class DyGFormerPredictor(nn.Module):
    """
    Simplified DyGFormer (Yu et al., NeurIPS 2023).

    Core DyGFormer ideas adapted for discrete snapshot graphs:

    1. Neighbor patch encoding:
       Original DyGFormer groups a node's temporal neighbors into fixed-size
       patches and applies transformer attention over them to capture
       long-range temporal dependencies.  Here we adapt this to snapshot
       graphs by applying scatter-based attention over 1-hop neighbors from
       the current snapshot edge_index — the same structural idea without
       continuous-time event sequences.

    2. Co-occurrence encoding:
       For a candidate link (u, v), DyGFormer computes a neighbor
       co-occurrence feature that captures how often u and v share common
       neighbors.  We implement this as a normalized Jaccard-style common-
       neighbor count, which can be computed efficiently from edge_index.

    3. Log time encoding (LTE from TAMI):
       We use LTE instead of the standard cosine encoding because our
       snapshot indices exhibit right-skewed spacing, matching LTE's
       design intent.

    Simplifications relative to the original:
    • Continuous-time temporal neighbor sequences → snapshot 1-hop neighbors
    • Full multi-head transformer over patches → single-layer scatter attention
    • Neighbor co-occurrence encoding scheme → Jaccard common-neighbor ratio
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_heads: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Log time encoder
        time_dim = max(8, hidden_dim // 8)
        self.time_enc = LogTimeEncoder(time_dim)
        self.time_dim = time_dim

        # GCN-style structural encoder
        self.gcn1 = nn.Linear(in_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)

        # Scatter attention over neighbors (the "patch attention" concept)
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Fuse structure + time
        self.fuse = nn.Linear(in_dim + time_dim, hidden_dim)
        self.fuse_norm = nn.LayerNorm(hidden_dim)

        # Co-occurrence encoding  (1 scalar → embedding)
        co_dim = max(4, hidden_dim // 8)
        self.co_dim  = co_dim
        self.co_proj = nn.Linear(1, co_dim)

        # Link decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + co_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def _gcn(self, x, edge_index, num_nodes):
        h = F.relu(self.gcn1(x))
        if edge_index.size(1) > 0:
            src, dst = edge_index
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.size(1)), h[src])
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
            h = F.relu(h + agg / deg.clamp(min=1).unsqueeze(1))
        return F.relu(self.gcn2(h))

    def _scatter_attention(self, h, edge_index, num_nodes):
        """Attention-weighted neighbor aggregation (patch attention concept)."""
        if edge_index.size(1) == 0:
            return h
        src, dst = edge_index
        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)
        # Edge-level attention scores
        scores = (Q[dst] * K[src]).sum(-1) * (self.hidden_dim ** -0.5)
        scores = scores - scores.max()                    # numerical stability
        exp_s  = torch.exp(scores)
        # Weighted aggregation
        agg = torch.zeros_like(V)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, V.size(1)),
                          exp_s.unsqueeze(1) * V[src])
        norm = torch.zeros(num_nodes, device=h.device)
        norm.scatter_add_(0, dst, exp_s)
        agg = agg / norm.clamp(min=1e-8).unsqueeze(1)
        return self.attn_norm(h + agg)

    def _co_occurrence(self, edge_index, src_nodes, dst_nodes, num_nodes):
        """Normalised common-neighbor count for each candidate link pair."""
        device = src_nodes.device
        if edge_index.size(1) == 0:
            z = torch.zeros(src_nodes.size(0), 1, device=device)
            return self.co_proj(z)

        # Build neighbor sets on CPU (fast for sparse graphs)
        ei = edge_index.cpu().numpy()
        nbr: dict = {}
        for s, d in zip(ei[0], ei[1]):
            nbr.setdefault(s, set()).add(d)

        src_np = src_nodes.cpu().numpy()
        dst_np = dst_nodes.cpu().numpy()
        co = []
        for u, v in zip(src_np, dst_np):
            nu = nbr.get(int(u), set())
            nv = nbr.get(int(v), set())
            denom = max(len(nu), len(nv), 1)
            co.append(len(nu & nv) / denom)

        co_t = torch.tensor(co, dtype=torch.float, device=device).unsqueeze(1)
        return self.co_proj(co_t)

    # ------------------------------------------------------------------
    def forward(self, x, edge_index, num_nodes, src_nodes, dst_nodes,
                snap_idx: int = 0, **kw):
        # Log time encoding
        t     = torch.full((num_nodes,), float(snap_idx), device=x.device)
        t_enc = self.time_enc(t)

        # GCN structural encoding + scatter attention (patch encoding)
        h = self._gcn(x, edge_index, num_nodes)
        h = self._scatter_attention(h, edge_index, num_nodes)

        # Fuse with time features
        h = h + self.fuse_norm(self.fuse(torch.cat([x, t_enc], dim=1)))

        # Co-occurrence feature for link pairs
        co = self._co_occurrence(edge_index, src_nodes, dst_nodes, num_nodes)

        return self.decoder(
            torch.cat([h[src_nodes], h[dst_nodes], co], dim=1)
        ).squeeze(1)

    @torch.no_grad()
    def predict(self, x, edge_index, num_nodes, src_nodes, dst_nodes, **kw):
        return torch.sigmoid(
            self.forward(x, edge_index, num_nodes, src_nodes, dst_nodes, **kw)
        ).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline 5 – TAMI-GraphMixer  (simplified)
# ─────────────────────────────────────────────────────────────────────────────
class TAMIPredictor(nn.Module):
    """
    TAMI framework wrapping GraphMixer (Yu et al., NeurIPS 2025).

    TAMI addresses heterogeneity in temporal interactions via two components:

    1. LTE (Log Time Encoding) — replaces cos(W·Δt) with cos(W·ln(1+Δt)).
       Proven to reduce skewness of Pareto-distributed temporal differences
       to exactly 2, making frequency parameters easier to learn.  Fully
       implemented here (not a simplification).

    2. LHA (Link History Aggregation) — maintains a memory of recent
       interactions for each target node pair so that infrequent connections
       are not forgotten when computing link probabilities.
       Simplified here: the memory tracks interaction frequency per pair
       (normalised count across training snapshots), encoded as a scalar
       feature added to the decoder.  Full LHA uses a learnable embedding
       r^τ_uv = γ·MLP([h_u;h_v]) + (1-γ)·r^{t1}_uv; our simplified
       version captures the same 'has this pair interacted before?' signal
       without the learnable edge embedding update.

    Base architecture: GraphMixer (MLP token-mixer + channel-mixer) with
    the standard TimeEncoder replaced by LTE.

    Call fit_lha_memory(train_graphs) once before training to populate
    the link-frequency dictionary from training snapshot edges.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 time_dim: int = 32):
        super().__init__()

        # ── LTE (fully faithful to TAMI paper) ───────────────────────────────
        self.lte = LogTimeEncoder(time_dim)

        # ── GraphMixer backbone ───────────────────────────────────────────────
        node_in = in_dim + time_dim
        self.token_mix = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim,     hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ── LHA memory ───────────────────────────────────────────────────────
        # link_freq[key] = count / n_train_snapshots  ∈ [0, 1]
        self._link_freq: dict = {}
        self._n_train: int    = 1

        # Encode LHA scalar into a small embedding
        lha_enc_dim = max(4, hidden_dim // 8)
        self.lha_enc     = nn.Linear(1, lha_enc_dim)
        self.lha_enc_dim = lha_enc_dim

        # ── Link decoder (with LHA feature) ──────────────────────────────────
        self.link_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + lha_enc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,      hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    def fit_lha_memory(self, train_graphs: list):
        """
        Pre-populate LHA memory from training snapshots.
        Call once before training; not differentiable (no-grad).
        """
        self._link_freq = {}
        self._n_train   = max(len(train_graphs), 1)
        for G in train_graphs:
            for u, v in G.edges():
                key = (min(u, v), max(u, v))
                self._link_freq[key] = self._link_freq.get(key, 0) + 1

    def reset_lha_memory(self):
        """Clear LHA memory (call between datasets)."""
        self._link_freq = {}
        self._n_train   = 1

    def _lha_features(self, src_nodes, dst_nodes) -> torch.Tensor:
        """Return normalised interaction-frequency feature for each pair."""
        device = src_nodes.device
        freqs = []
        for u, v in zip(src_nodes.cpu().numpy(), dst_nodes.cpu().numpy()):
            key  = (min(int(u), int(v)), max(int(u), int(v)))
            freq = self._link_freq.get(key, 0) / self._n_train
            freqs.append(freq)
        f = torch.tensor(freqs, dtype=torch.float, device=device).unsqueeze(1)
        return self.lha_enc(f)                     # (N_pairs, lha_enc_dim)

    # ------------------------------------------------------------------
    def forward(self, x, edge_index, num_nodes, src_nodes, dst_nodes,
                snap_idx: int = 0, **kw):
        # LTE: log time encoding (replaces standard TimeEncoder)
        t     = torch.full((num_nodes,), float(snap_idx), device=x.device)
        t_enc = self.lte(t)

        # GraphMixer with LTE
        h = self.norm1(self.token_mix(torch.cat([x, t_enc], dim=1)))
        h = self.norm2(h + self.channel_mix(h))

        # LHA feature for target pairs
        lha = self._lha_features(src_nodes, dst_nodes)

        return self.link_mlp(
            torch.cat([h[src_nodes], h[dst_nodes], lha], dim=1)
        ).squeeze(1)

    @torch.no_grad()
    def predict(self, x, edge_index, num_nodes, src_nodes, dst_nodes, **kw):
        return torch.sigmoid(
            self.forward(x, edge_index, num_nodes, src_nodes, dst_nodes, **kw)
        ).cpu().numpy()