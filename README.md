# P53BindML: An ML-Guided SBVS Framework for Prioritizing Compounds Targeting Mutant p53 (R248H/R273H)

Machine learning framework for structure-based virtual screening (SBVS), designed to prioritize small-molecule compounds targeting the p53 hotspot mutants **R248H** and **R273H**. The framework benchmarks six machine learning models — from classical classifiers to deep learning and graph neural networks — trained on protein-ligand interaction fingerprints and molecular graph representations derived from docking poses.

## Overview

This repository implements a complete pipeline that goes from docked ligand-receptor complexes to trained, evaluated, and saved classification models, ready to be used for virtual screening of new compound libraries against p53 mutants.

The framework compares two feature representation strategies:

- **PLEC (Protein-Ligand Extended Connectivity) fingerprints** — encode the interaction environment between ligand and receptor atoms within a defined distance cutoff, generated with [ODDT](https://github.com/oddt/oddt).
- **Molecular graphs (ConvMol)** — atom-level features and bond connectivity per ligand, generated with [DeepChem](https://github.com/deepchem/deepchem) and converted to [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) `Data` objects for graph neural network training.

Six models are trained and evaluated under a consistent protocol:

| Model | Feature type | Library |
|---|---|---|
| Support Vector Machine (SVM, RBF kernel) | PLEC | scikit-learn |
| Random Forest (RF) | PLEC | scikit-learn |
| XGBoost (XGB) | PLEC | xgboost |
| Multi-Layer Perceptron (ANN) | PLEC | scikit-learn |
| Deep Neural Network (DNN) | PLEC | TensorFlow/Keras |
| Graph Convolutional Network (GCN) | ConvMol graphs | PyTorch Geometric |

## Evaluation protocol

Each model is trained and evaluated over **10 repeated runs**, each consisting of:

1. **5-fold stratified cross-validation** on the training set (reseeded per run) to assess stability of ROC-AUC and PR-AUC.
2. **Final training** on the full training set.
3. **Held-out test set evaluation** using the final model, reporting the same metrics.

## Methodology and references

The machine learning models used in this framework are based on the following references:

This methodology and code are based on the Nature Protocols paper:

> Tran-Nguyen, V. K., Junaid, M., Simeon, S., & Ballester, P. J. (2023). *A practical guide to machine-learning scoring for structure-based virtual screening.* Nature Protocols. [vktrannguyen/MLSF-protocol](https://github.com/vktrannguyen/MLSF-protocol)

We have made some changes to hyperparameter optimization and added cross-validation for each model.

The graph convolutional network (GCN) architecture and training procedure were additionally informed by:

> Gardeina/TSSF-GCN — [Graph convolutional neural networks improved target-specific scoring functions for cGAS and kRAS in virtual screening](https://www.sciencedirect.com/science/article/pii/S2001037025001886). [GitHub repository](https://github.com/Gardeina/TSSF-GCN)

A simialr framework also builds on our previous work applying machine-learning-guided virtual screening to protein-protein interaction targets:

> Castillo Tarazona MY and Miscione GP (2026) Enhancing structure-based virtual screening of MDM2–p53 inhibitors: a benchmark of machine learning vs. traditional docking scoring functions. *Front. Drug Discov.* 5:1731262. doi: [10.3389/fddsv.2025.1731262](https://doi.org/10.3389/fddsv.2025.1731262). [GitHub repository](https://github.com/MarciaC18/Identification-of-MDM2-P53-Inhibitors-Using-Machine-Learning-Guided-Screening)

## Repository structure

```
├── data/
│   ├── data-strategy1/
│   │   ├── train/
│   │   │   ├── data/train_data.csv        # mol_name, receptor, activity (Active/Inactive)
│   │   │   ├── ligands/                   # Docked ligand poses (.mol2), training set
│   │   │   └── receptor/                  # Receptor structures (.mol2)
│   │   └── test/
│   │       ├── data/test_data.csv
│   │       ├── ligands/
│   │       └── receptor/
│   ├── data-strategy2/
│   │   ├── train/
│   │   └── test/
│   └── data-strategy3/
│       ├── train/
│       └── test/
├── Models/
│   ├── Step1-featurize-ligands-graphs.py  # PLEC / ConvMol featurization from docked ligands
│   ├── Step2-build-graphs.py              # Build PyTorch Geometric graphs from ConvMol features
│   ├── SVM.py
│   ├── RF.py
│   ├── XGB.py
│   ├── ANN.py
│   ├── DNN.py
│   └── GNN.py
└── README.md
```

## Requirements

```
python >= 3.9
rdkit
oddt
deepchem
scikit-learn
xgboost
tensorflow
torch
torch-geometric
pandas
numpy
joblib
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

Before running the pipeline, we suggest configuring the `protocol-env` environment, using the `protocol-env.yml` file available in [vktrannguyen/MLSF-protocol](https://github.com/vktrannguyen/MLSF-protocol):
```bash
conda env create -f protocol-env.yml
conda activate protocol-env
```

## Usage

**1. Featurize ligands** (from docked `.mol2` poses):
```bash
python Models/Step1-featurize-ligands-graphs.py
python Models/Step2-build-graphs.py
```

**2. Train and evaluate a model** (example: XGBoost):
```bash
python Models/XGB.py
```

Each script produces per-fold CV metrics, per-run test predictions, an aggregated summary CSV across the 10 runs, and a saved final model (plus PLEC parameters) ready for downstream virtual screening.

## Outputs

For every model, results include:
- `Active_Prob`, `Inactive_Prob`, `Predicted_Class`, `Real_Class` per test compound
- ROC-AUC and PR-AUC, for both cross-validation and the held-out test set
- A saved, ready-to-use model file for virtual screening of new compound libraries

## Citation

If you use this framework, please cite:

> A manuscript describing this framework is in preparation. In the meantime, if you use this code, please cite the repository directly:

> Castillo Tarazona, M. (2026). *P53BindML: An ML-Guided SBVS Framework for Prioritizing Compounds Targeting Mutant p53 (R248H/R273H)* [Software]. GitHub. https://github.com/MarciaC18/P53BindML-An-ML-Guided-SBVS-Framework-for-Prioritizing-Compounds-Targeting-Mutant-p53-R248H-R273H

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Marcia Castillo — m.castillot@uniandes.edu.co
Universidad de los Andes, Computational Bioorganic Chemistry, Bogotá, Colombia
