
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

## Running the Experiments

### 1. Dataset Preprocessing

#### Reddit

```bash
python process_reddit_hyperlinks.py soc-redditHyperlinks-body.tsv --max_nodes 35776 --n_snapshots 39 --window_days 30 --out reddit_30day.pkl
```
### AskUbuntu

```bash
python process_sx_temporal.py sx-askubuntu.txt --max_nodes 100000 --n_snapshots 39 --window_days 30 --out askubuntu_30day.pkl
```

### SuperUser

```bash
python process_sx_temporal.py sx-superuser.txt --max_nodes 100000 --n_snapshots 39 --window_days 30 --out superuser_30day.pkl
```

## Main Link Prediction Experiments

### Reddit

```bash
python train_link_pred.py --data_path reddit_30day.pkl --dataset_name Reddit --epochs 100
```
### AskUbuntu

```bash
python train_link_pred.py --data_path askubuntu_30day.pkl --dataset_name AskUbuntu --epochs 100
```

### SuperUser

```bash
python train_link_pred.py --data_path superuser_30day.pkl --dataset_name SuperUser --epochs 100
```
## Ablation Studies

### Reddit

```bash
python ablation.py --data_path reddit_30day.pkl --dataset_name Reddit --epochs 100
```

### AskUbuntu

```bash
python ablation.py --data_path askubuntu_30day.pkl --dataset_name AskUbuntu --epochs 100
```

### SuperUser

```bash
python ablation.py --data_path superuser_30day.pkl --dataset_name SuperUser --epochs 100
```

## Multi-Step Rollout Experiments

### Reddit

```bash
python rollout.py --data_path reddit_30day.pkl --dataset_name Reddit --epochs 100
```

### AskUbuntu

```bash
python rollout.py --data_path askubuntu_30day.pkl --dataset_name AskUbuntu --epochs 100
```

### SuperUser

```bash
python rollout.py --data_path superuser_30day.pkl --dataset_name superuser --epochs 100
```

