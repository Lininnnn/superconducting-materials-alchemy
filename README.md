# superconducting-materials-alchemy

# HGTC-Net

HGTC-Net (Hierarchical Graph Tc Network) is a hierarchical classification--regression framework for superconducting critical temperature (\(T_c\)) prediction based on crystal graph learning and superconducting category priors.

The framework combines:

- **HSC-XGB**: Hierarchical Statistical Classification with XGBoost
- **PriorGNN**: Prior-guided Graph Neural Network

HGTC-Net integrates crystal graph topology, physicochemical statistical descriptors, and superconducting category priors to improve superconducting material classification and superconducting critical temperature prediction.

---

# Framework Overview

The overall workflow of HGTC-Net includes:

1. Crystal structure parsing from CIF files
2. Crystal graph construction
3. Graph-level statistical descriptor generation
4. Hierarchical superconducting classification
5. Semantic prior generation
6. Graph neural network regression
7. Superconducting critical temperature prediction

---

# Key Features

- Crystal graph representation learning
- Hierarchical superconducting family classification
- Semantic prior guided regression
- Residual GINEConv architecture
- Multi-scale graph pooling
- Physically interpretable node and edge features
- External database screening capability

---

# Dataset

This work mainly uses the public superconducting materials database:

- 3DSC
- Materials Project (MP)
- ICSD

The dataset includes:

- Crystal structures in CIF format
- Experimental superconducting critical temperatures
- Cu-based superconductors
- Fe-based superconductors
- Hydride superconductors
- Other superconducting systems
- Non-superconducting materials

---

# Crystal Graph Construction

Each crystal structure is converted into a graph representation:

```math
G=(V,E)
```

where:

- \(V\) represents atomic nodes
- \(E\) represents neighboring atomic connections

Neighbor searching is performed using:

```python
struct.get_all_neighbors(r=6.0)
```

---

# Node Features

Each atom is represented by an 8-dimensional feature vector including:

- Atomic mass
- Electronegativity
- Atomic radius
- Group number
- Period number
- Metallicity
- Atomic number
- Mendeleev number

---

# Edge Features

Edge descriptors include:

- Interatomic distance
- Electronegativity difference

For PriorGNN regression, simplified edge features are used:

```math
e_{ij}^{reg}=
\left[
\frac{1}{d_{ij}^2},
\Delta EN_{ij}
\right]
```

---

# HSC-XGB

HSC-XGB performs hierarchical superconducting classification using graph-level statistical descriptors.

## Binary Classification

Predicts:

- Superconducting
- Non-superconducting

```math
P_{bin}=
[P_{SC},P_{Non-SC}]
```

## Multi-class Classification

Further classifies superconductors into:

- Cu-based superconductors
- Fe-based superconductors
- Other superconducting systems

```math
P_{multi}=
[P_{Cu},P_{Fe},P_{Other}]
```

The final semantic prior descriptor is:

```math
P_{class}
=
[P_{SC},P_{Non-SC},P_{Cu},P_{Fe},P_{Other}]
```

---

# PriorGNN

PriorGNN incorporates category-aware semantic priors into graph neural network regression.

Main components include:

- Semantic gating mechanism
- Residual GINEConv layers
- Multi-scale pooling
- MLP regression head

The predicted superconducting critical temperature is restored using:

```math
T_c=e^{\hat y}-1
```

---

# Experimental Results

## Classification Performance

| Category | Accuracy |
|---|---|
| Cu-based SC | 0.90 |
| Fe-based SC | 0.93 |
| Other SC | 0.95 |
| Non-SC | 0.81 |

## Regression Performance

| Dataset | \(R^2\) | RMSE (K) | MAE (K) |
|---|---|---|---|
| Train | 0.9065 | 5.7814 | 2.6034 |
| Test | 0.8809 | 6.6113 | 3.0574 |

---

# External Screening

HGTC-Net was further evaluated on external crystal structures collected from the GNoME database.

The framework successfully identified large numbers of potential superconducting candidates, including:

- Cu-based superconductors
- Fe-based superconductors
- Hydrogen-rich superconductors

Several predicted candidates exhibit high predicted superconducting critical temperatures approaching \(100~\mathrm{K}\) and above.

---

# Requirements

Recommended environment:

```bash
Python >= 3.9
PyTorch
PyTorch Geometric
pymatgen
xgboost
scikit-learn
numpy
pandas
matplotlib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Example training command:

```bash
python train.py
```

---

# Inference

Example inference command:

```bash
python predict.py
```

---

# Project Structure

```text
HGTC-Net/
│
├── data/
├── models/
│   ├── hsc_xgb.py
│   ├── priorgnn.py
│
├── utils/
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

# License

This project is released under the MIT License.
