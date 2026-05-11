# NetT-learner

A package-style implementation of **NetT-learner**, a meta-learning framework for estimating causal effects under **network interference**.

This repository provides:

* A reproducible **simulation framework**
* A modular **Python package (`nett_learner`)**
* A **real-world dataset** based on U.S. counties
* Implementations of:

  * `NetTLinear` (linear outcome model)
  * `NetTGCN` (graph neural network model)

---

# 🔬 Method Overview

NetT-learner estimates:

* **Direct effects**: effect of a unit’s own treatment
* **Peer effects (spillover effects)**: effect of neighbors’ treatments

The framework consists of two stages:

1. **Stage 1 (Node-level estimation)**
   Learn outcome models:

   * μ₁(x, g)
   * μ₀(x, g)

   Then compute node-level effects.

2. **Stage 2 (Kernel smoothing)**
   Estimate conditional effects.

---

# Real Data

This repository includes a real-world dataset for studying **network interference across U.S. counties**.

## Data Files

* `final_covariates_corr_lt_0p8.csv`
  Preprocessed county-level covariates (after correlation filtering).  
  These variables are constructed from multiple public data sources, including:
  - CDC PLACES dataset (health-related indicators): https://www.cdc.gov/places  
  - CDC/ATSDR Social Vulnerability Index (SVI): https://www.atsdr.cdc.gov/placeandhealth/svi/index.html  

* `final_outcome.csv`
  Outcome variable (e.g., infection measure)

* `final_vax1.csv`
  First-dose vaccination rates

* `final_vax2.csv`
  Second-dose vaccination rates

* `vax_common.csv`
  Harmonized vaccination dataset

* `network_common.csv`
  County-level mobility network (edges represent movement flows)

* `node_effects_all_back2logY.csv`
  Node-level estimated effects (direct + peer, log scale)

---

## Data Description

* **Unit**: U.S. county (FIPS code)
* **Network**: mobility-based connections between counties
* **Treatment**: vaccination coverage (thresholded)
* **Outcome**: log-transformed infection-related measure
* **Covariates**:

  * demographic variables
  * socioeconomic indicators
  * health-related variables

---

## Network Construction

The network is constructed from mobility data:

* Nodes: counties
* Edges: presence of movement between counties

Example:

```python
import networkx as nx
import pandas as pd

network = pd.read_csv("network_common.csv")

G = nx.DiGraph()
for u, v in zip(network["geoid_o"], network["geoid_d"]):
    G.add_edge(str(u), str(v))
```

---

## Real Data Workflow

### Step 1: Load data

```python
import pandas as pd

cov = pd.read_csv("final_covariates_corr_lt_0p8.csv")
outcome = pd.read_csv("final_outcome.csv")
vax1 = pd.read_csv("final_vax1.csv")
vax2 = pd.read_csv("final_vax2.csv")
network = pd.read_csv("network_common.csv")
```

---

### Step 2: Construct variables

You need to build:

* `X`: node covariates
* `X_neighbor`: neighbor-aggregated covariates
* `Z`: treatment indicator
* `G`: exposure (treated neighbor intensity)

---

### Step 3: Fit model

```python
model = NetTGCN(kernel="kr_rbf").fit(data)

results = model.estimate_effects(
    data,
    X_cate=data.X_raw[:, 0].reshape(-1, 1)
)
```

---

## 📈 Outputs

The model returns:

* `direct_node`: node-level direct effects
* `peer_node`: node-level peer effects
* `direct_cate`: smoothed direct effects
* `peer_cate`: smoothed peer effects

---

## 🧩 Interpretation

* **Direct effect**
  Effect of a county’s own treatment

* **Peer effect**
  Effect of neighboring counties’ treatments

* **CATE**
  Heterogeneity of effects across covariates

---

## ⚠️ Notes

* Kernel smoothing is applied in a second stage
* Multi-dimensional `X_cate` is supported
* For visualization, use 1D features

---

# 📁 Project Structure

```text
NetT-learner/
├── src/nett_learner/
│   ├── data.py
│   ├── linear.py
│   ├── gcn.py
│   ├── smoothing.py
│   └── utils.py
├── example.py
├── README.md
└── pyproject.toml
```

---

# 🔁 Reproducibility

* Python ≥ 3.10
* Recommended:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python example.py
```

---

# 📌 Summary

This repository provides:

* A modular implementation of NetT-learner
* A simulation framework for network interference
* A real-world application on U.S. county data

---

# 📎 Citation

(Coming soon — add your paper here)
