# Link Prediction in Dynamic Social Networks with a World Model

### Efficiency, Imagined Rollouts, and Historical-Negative Evaluation

This repository contains the implementation and experimental code for our
study of **link prediction in dynamic social networks using a snapshot-level
World Model**.

The study evaluates whether a lightweight graph-based model can provide
competitive link-prediction performance while reducing reported runtime and
whether its predictions remain usable when previously observed graph states
are replaced by **imagined states generated through multi-step rollouts**.

We evaluate the proposed World Model on dynamic graphs derived from:

- Reddit
- AskUbuntu
- SuperUser

and compare it with simplified implementations of:

- DyRep
- GraphMixer
- DyGFormer
- TAMI

---

## Overview

Dynamic social networks evolve over time, making link prediction a temporal
forecasting problem rather than a static graph-completion task.

Our World Model uses a shared **two-layer GCN encoder** followed by a
learned **MLP edge decoder**.

The model operates on graph snapshots:

```text
    G_t
     │
     ├── Structural node features
     │
     ▼
┌─────────────────────┐
│     2-Layer GCN     │
│  Graph Encoder      │
└─────────────────────┘
     │
     ▼
Node embeddings
     │
 ────────────────┐
 │               │
 ▼               ▼
z_u             z_v
 │               │
 └──────┬────────┘
        ▼
┌─────────────────────┐
│      MLP Decoder    │
└─────────────────────┘
        │
        ▼
   Link probability
```

## Repository Structure

```text
ACM_Ireland_Link_Prediction/
│
├── Datasets/
│
├── worldmodel.py
│   └── World Model / GCN components
│
├── link_prediction.py
│   └── Link-prediction models, baselines,
│       metrics and negative sampling
│
├── train_link_pred.py
│   └── Main link-prediction experiments
│
├── rollout.py
│   └── Multi-step imagined rollout experiments
│
├── ablation.py
│   └── Ablation experiments
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Ansh911/ACM_Ireland_Link_Prediction.git
cd ACM_Ireland_Link_Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

