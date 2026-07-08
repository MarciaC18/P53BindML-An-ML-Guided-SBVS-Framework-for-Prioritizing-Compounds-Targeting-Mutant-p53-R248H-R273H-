# =========================================================
# STEP 2: BUILD PYTORCH GEOMETRIC GRAPHS
# =========================================================
# This script loads the ligand features produced in Step 1 (adjacency
# matrices and atom features per molecule), matches each ligand to its
# activity label using the dataset's CSV file, and builds a PyTorch
# Geometric graph object for every ligand. All graphs for a dataset are
# saved together as a single .pt file, ready to be used for GNN training.
# =========================================================

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path


def build_graphs(npz_file, csv_file, output_pt):
    """
    Build a list of PyTorch Geometric graphs from featurized ligands and
    their activity labels, then save the list to output_pt.

    Steps:
      1. Load the adjacency matrices, atom features, and ligand names
         produced by Step 1.
      2. Load the activity labels (Active/Inactive) from the dataset CSV,
         indexed by ligand name.
      3. For each ligand, build a torch_geometric.data.Data object
         containing its node features, edge connectivity, and label.
      4. Save the full list of graphs to disk.
    """
    npz_file = Path(npz_file)
    csv_file = Path(csv_file)
    output_pt = Path(output_pt)

    # Make sure the output folder exists before saving.
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load featurized ligands (Step 1 output)
    # -----------------------------
    data = np.load(npz_file, allow_pickle=True)

    adj_mats = data["adjacency_matrices"]
    node_feats = data["node_features"]
    ligand_names = data["ligand_names"]

    # -----------------------------
    # Load activity labels
    # -----------------------------
    df = pd.read_csv(csv_file, sep=';')
    # Maps each ligand name to its activity label ("Active"/"Inactive"),
    # used to attach the correct target value to each graph below.
    label_map = dict(zip(df.mol_name, df.activity))

    graphs = []

    # -----------------------------
    # Build one PyTorch Geometric graph per ligand
    # -----------------------------
    for adj, x, name in zip(adj_mats, node_feats, ligand_names):
        if name not in label_map:
            # Ligand has no matching activity label in the CSV; skip it.
            continue

        # edge_index encodes graph connectivity as a [2, num_edges] tensor:
        # each column is a (source_atom, target_atom) bonded pair, derived
        # directly from the nonzero entries of the adjacency matrix.
        edge_index = torch.tensor(
            np.array(adj).nonzero(),
            dtype=torch.long
        )

        # Per-atom feature matrix for this molecule.
        x = torch.tensor(x, dtype=torch.float)

        # Binary activity label: 1 for Active, 0 for Inactive.
        y = torch.tensor(
            [1 if label_map[name] == "Active" else 0],
            dtype=torch.float
        )

        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    # -----------------------------
    # Save the full list of graphs
    # -----------------------------
    torch.save(graphs, output_pt)
    print(f"✅ Saved {len(graphs)} graphs to {output_pt}")


if __name__ == "__main__":
    build_graphs(
        "features/train_ConvMol.npz",
        "train/data/train_data.csv",
        "gnn_input/train_graphs.pt"
    )

    build_graphs(
        "features/test_ConvMol.npz",
        "test/data/test_data.csv",
        "gnn_input/test_graphs.pt"
    )
