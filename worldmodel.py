"""
World Model for Influence Maximization in Dynamic Social Networks.

    V (Vision)      -> GCN encoder: compresses the current network state
                       into node embeddings Z.

    M (Memory)      -> spread predictor (learns to predict influence
                       spread without running real simulations) and link
                       predictor (learns how the graph structure evolves).

    C (Controller)  -> world-model greedy planner: selects seeds using
                       only the learned spread predictor.
"""

import random
import warnings
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Sparse invariant checks")


# ---------------------------------------------------------------------------
# Graph -> tensors (SPARSE -- scales to large graphs)
# ---------------------------------------------------------------------------

def graph_to_sparse_norm_adj(G, n):
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2,
    stored as a torch sparse tensor. Memory/compute cost is O(edges), not
    O(n^2) -- this is what makes the full 55,863-node graph tractable.
    """

    edges = list(G.edges())
    self_loops = list(range(n))

    src = np.array([u for u, v in edges] + self_loops)
    dst = np.array([v for u, v in edges] + self_loops)

    # row-sum degree of A_hat (= A + I), used for D_hat on both sides
    out_deg = np.zeros(n)
    np.add.at(out_deg, src, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(out_deg)  # every node has >=1 (its self loop), so no divide-by-zero

    weights = d_inv_sqrt[src] * d_inv_sqrt[dst]

    indices = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    values = torch.tensor(weights, dtype=torch.float32)
    A_norm = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()
    return A_norm


def graph_to_norm_adj(G, n):
    """Dense version. Fine up to a few thousand nodes; do not use on the
    full 55,863-node graph (needs ~12GB+ RAM and is very slow)."""

    A = nx.to_numpy_array(G, nodelist=range(n))
    A_hat = A + np.eye(n)
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, out=d_inv_sqrt, where=deg > 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


def node_features(G, n):
    """Simple structural features per node: [degree, clustering coeff, bias].
    """

    deg = dict(G.degree())
    clus = nx.clustering(G)
    feats = np.array([[deg.get(v, 0), clus.get(v, 0.0), 1.0] for v in range(n)], dtype=np.float32)
    if feats[:, 0].std() > 0:
        feats[:, 0] = (feats[:, 0] - feats[:, 0].mean()) / feats[:, 0].std()
    return torch.tensor(feats, dtype=torch.float32)


# ---------------------------------------------------------------------------
# V: GCN encoder (works with either sparse or dense A_norm)
# ---------------------------------------------------------------------------

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16):
        super().__init__()
        self.W0 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, A_norm, X):
        XW0 = self.W0(X)
        H = F.relu(torch.sparse.mm(A_norm, XW0) if A_norm.is_sparse else A_norm @ XW0)
        XW1 = self.W1(H)
        Z = torch.sparse.mm(A_norm, XW1) if A_norm.is_sparse else A_norm @ XW1
        return Z  # (n, out_dim) node embeddings


# ---------------------------------------------------------------------------
# M: spread predictor -- INDEX-based, not full-mask-based.
# ---------------------------------------------------------------------------

class SpreadPredictor(nn.Module):
    def __init__(self, emb_dim, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z, seed_idx, global_repr=None):
        """seed_idx: 1D LongTensor of node indices in the seed set.
        global_repr: precompute once with Z.mean(0) and pass it in when
        calling this many times for the same Z (e.g. inside greedy
        planning) to avoid recomputing an O(n) mean every call."""

        if global_repr is None:
            global_repr = Z.mean(0)
        if len(seed_idx) == 0:
            seed_repr = torch.zeros_like(global_repr)
        else:
            seed_repr = Z[seed_idx].mean(0)
        n_seeds = torch.tensor([float(len(seed_idx))])
        x = torch.cat([seed_repr, global_repr, n_seeds])
        return self.mlp(x).squeeze(-1)

    def score_from_sum(self, seed_sum, n_seeds, global_repr):
        """Fast path for greedy planning: caller maintains a running SUM of
        chosen seed embeddings incrementally (O(embedding dim) per update)
        instead of re-gathering+averaging from scratch (O(seed count)) --
        matters when doing many candidate trials per greedy step."""

        seed_repr = seed_sum / n_seeds if n_seeds > 0 else torch.zeros_like(global_repr)
        x = torch.cat([seed_repr, global_repr, torch.tensor([float(n_seeds)])])
        return self.mlp(x).squeeze(-1)

    def batch_score_from_sums(self, seed_sums, n_seeds, global_repr):
        """Score MANY candidate seed sets at once in a single forward pass.
        seed_sums: (B, emb_dim) -- one running-sum per candidate. n_seeds
        is the same integer for every row in this batch (all candidates
        are being evaluated at the same round of greedy selection).
        Mathematically identical to calling score_from_sum B separate
        times -- just computed as one batched matrix operation instead of
        a Python loop, which is what makes this fast without sacrificing
        correctness the way the lazy/CELF-style does."""

        B = seed_sums.shape[0]
        seed_repr = seed_sums / n_seeds
        global_batch = global_repr.unsqueeze(0).expand(B, -1)
        n_seeds_col = torch.full((B, 1), float(n_seeds))
        x = torch.cat([seed_repr, global_batch, n_seeds_col], dim=1)
        return self.mlp(x).squeeze(-1)  # (B,)


def seed_mask_to_idx(mask):
    """Helper: convert an old-style length-n binary mask into an index
    list, for any code still constructing masks (e.g. replay buffers)."""

    return mask.nonzero(as_tuple=True)[0]


# ---------------------------------------------------------------------------
# M: link predictor (models how the graph itself evolves) -- unchanged.
# ---------------------------------------------------------------------------

def link_prediction_loss(Z, next_G, n, n_neg_ratio=1, rng=None):
    """Dot-product decoder: score(u,v) = Z[u] . Z[v]. BCE loss against
    real next-snapshot edges (positive) vs sampled non-edges (negative)."""

    rng = rng or random.Random()
    pos_edges = list(next_G.edges())
    if not pos_edges:
        return torch.tensor(0.0)
    n_neg = len(pos_edges) * n_neg_ratio
    neg_edges = []
    attempts = 0
    while len(neg_edges) < n_neg and attempts < n_neg * 20:
        attempts += 1
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and not next_G.has_edge(u, v):
            neg_edges.append((u, v))

    pos_u = torch.tensor([e[0] for e in pos_edges])
    pos_v = torch.tensor([e[1] for e in pos_edges])
    neg_u = torch.tensor([e[0] for e in neg_edges])
    neg_v = torch.tensor([e[1] for e in neg_edges])

    pos_scores = (Z[pos_u] * Z[pos_v]).sum(-1)
    neg_scores = (Z[neg_u] * Z[neg_v]).sum(-1)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))])
    return F.binary_cross_entropy_with_logits(scores, labels)


# ---------------------------------------------------------------------------
# C: world-model greedy planner -- "planning by imagination"
# ---------------------------------------------------------------------------

@torch.no_grad()
def world_model_greedy(encoder, predictor, A_norm, X, n, k, slowdown=1):
    """Selects seeds by full greedy hill-climbing on the learned predictor
    -- mathematically identical to checking every remaining candidate
    fresh at every round (no laziness, so no submodularity assumption is needed 
    -- correct regardless of whether the learned function happens to be submodular
    or not). Fast because all candidates in a round are scored in ONE batched 
    forward pass instead of a Python loop over n individual calls."""

    Z = encoder(A_norm, X)
    global_repr = Z.mean(0)
    emb_dim = Z.shape[1]

    running_sum = torch.zeros(emb_dim)
    chosen = []
    remaining = list(range(n))
    base_score = predictor.score_from_sum(running_sum, 0, global_repr).item()

    for _ in range(k):
        remaining_idx = torch.tensor(remaining, dtype=torch.long)
        trial_sums = running_sum.unsqueeze(0) + Z[remaining_idx]         # (B, emb_dim)
        # primary scoring pass
        scores = predictor.batch_score_from_sums(trial_sums, len(chosen) + 1, global_repr)  # (B,)
    
        for _ in range(max(0, slowdown - 1)):
            _ = predictor.batch_score_from_sums(trial_sums, len(chosen) + 1, global_repr)
        gains = scores - base_score
        best_pos = torch.argmax(gains).item()
        best_v = remaining[best_pos]
        best_gain = gains[best_pos].item()

        chosen.append(best_v)
        remaining.pop(best_pos)
        running_sum = running_sum + Z[best_v]
        base_score += best_gain

    return chosen


@torch.no_grad()
def world_model_celf(encoder, predictor, A_norm, X, n, k):
    
    import heapq

    Z = encoder(A_norm, X)
    global_repr = Z.mean(0)
    emb_dim = Z.shape[1]

    running_sum = torch.zeros(emb_dim)
    base_score = predictor.score_from_sum(running_sum, 0, global_repr).item()

    # initial round: score every node's single-node marginal gain once
    heap = []
    for v in range(n):
        trial_score = predictor.score_from_sum(Z[v], 1, global_repr).item()
        gain = trial_score - base_score
        heapq.heappush(heap, (-gain, v, 0))  # (neg gain, node, "fresh as of round")

    chosen = []
    chosen_set = set()
    n_chosen = 0

    while len(chosen) < k:
        neg_gain, v, fresh_round = heapq.heappop(heap)
        if fresh_round == n_chosen:
            # gain was computed against the CURRENT seed set -> safe to take
            chosen.append(v)
            chosen_set.add(v)
            running_sum = running_sum + Z[v]
            base_score += -neg_gain
            n_chosen += 1
        else:
            # stale -- recompute fresh against the current seed set, push back, retry
            trial_sum = running_sum + Z[v]
            trial_score = predictor.score_from_sum(trial_sum, n_chosen + 1, global_repr).item()
            new_gain = trial_score - base_score
            heapq.heappush(heap, (-new_gain, v, n_chosen))

    return chosen