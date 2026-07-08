# =========================================================
# STEP 1: LIGAND FEATURIZATION (ConvMol graphs)
# =========================================================
# This script reads all ligand structures (.mol2 files) from a dataset
# folder, converts each one into a molecular graph representation using
# DeepChem's ConvMolFeaturizer, and saves the resulting adjacency matrices
# and atom-level features to a single .npz file for later use.
# =========================================================

import glob
import numpy as np
import deepchem as dc
from rdkit import Chem
from pathlib import Path


def featurize_ligands(folder_path, output_file):
    """
    Featurize all ligands in a folder as ConvMol graphs and save the result.

    For each .mol2 file found in folder_path:
      1. Parse it into an RDKit molecule object.
      2. Convert it into a ConvMol graph (atom features + adjacency list).
      3. Build a dense adjacency matrix from the adjacency list.
    All results are collected and saved together in a single .npz archive.
    """
    # Collect and sort all ligand files so the processing order is stable
    # and reproducible across runs.
    ligand_files = glob.glob(str(Path(folder_path) / "*.mol2"))
    ligand_files.sort()

    rdkit_mols = []
    ligand_names = []

    # -----------------------------
    # Load ligands with RDKit
    # -----------------------------
    for f in ligand_files:
        # sanitize=True runs RDKit's standard chemical validity checks;
        # removeHs=False keeps explicit hydrogens in the molecular graph.
        mol = Chem.MolFromMol2File(f, sanitize=True, removeHs=False)
        if mol is not None:
            rdkit_mols.append(mol)
            # Keep track of the ligand's file name (without extension) so
            # each graph can later be matched back to its activity label.
            ligand_names.append(Path(f).stem)
        else:
            print(f"[WARNING] Could not read: {f}")

    print(f"[{folder_path}] Ligands loaded: {len(rdkit_mols)}")

    # -----------------------------
    # ConvMol featurization
    # -----------------------------
    # ConvMolFeaturizer computes, for each molecule, a set of per-atom
    # features and a neighbor list describing which atoms are bonded to
    # which.
    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer.featurize(rdkit_mols)

    adjacency_matrices = []
    node_features = []

    for i, convmol in enumerate(X):
        # Neighbor-list representation: for each atom index, the list of
        # atom indices it is bonded to.
        adj_list = convmol.get_adjacency_list()
        n_atoms = len(adj_list)

        # Convert the neighbor list into a dense n_atoms x n_atoms binary
        # adjacency matrix (1 where two atoms are bonded, 0 otherwise).
        adj = np.zeros((n_atoms, n_atoms), dtype=int)
        for j, neighs in enumerate(adj_list):
            for k in neighs:
                adj[j, k] = 1

        adjacency_matrices.append(adj)
        # Per-atom feature vectors (atom type, degree, hybridization, etc.,
        # as defined by ConvMolFeaturizer).
        node_features.append(convmol.get_atom_features())

        print(f"Mol {i}: adj {adj.shape}, node_feat {convmol.get_atom_features().shape}")

    # -----------------------------
    # Save to .npz
    # -----------------------------
    # Adjacency matrices, atom features, and ligand names are saved
    # together so downstream scripts can rebuild each molecular graph and
    # match it to its corresponding label by name.
    np.savez(
        output_file,
        adjacency_matrices=adjacency_matrices,
        node_features=node_features,
        ligand_names=ligand_names
    )

    print(f"Saved: {output_file}\n")


# -----------------------------
# Featurize both the training and test sets
# -----------------------------
featurize_ligands("train/ligands", "features/train_ConvMol.npz")
featurize_ligands("test/ligands", "features/test_ConvMol.npz")
