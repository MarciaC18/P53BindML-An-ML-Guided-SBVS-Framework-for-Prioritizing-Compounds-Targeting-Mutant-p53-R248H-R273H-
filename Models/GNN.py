import torch
import numpy as np
import pandas as pd
import os

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, matthews_corrcoef
import joblib

# =====================================================
# NOTE
# =====================================================
# GCN has genuine per-run stochasticity: layer
# weight initialization, Dropout masking, and DataLoader mini-batch shuffling
# are all governed by PyTorch's/NumPy's global random state, reseeded per run
# via torch.manual_seed/np.random.seed. Each run therefore trains a
# genuinely different set of weights, not just a different CV split. The
# outer StratifiedKFold split is also reseeded per run.
# =====================================================


# =====================================================
# LOAD GRAPHS
# =====================================================
train_graphs = torch.load("gnn_input/train_graphs.pt")
test_graphs  = torch.load("gnn_input/test_graphs.pt")

train_labels = np.array([g.y.item() for g in train_graphs])
test_labels  = np.array([g.y.item() for g in test_graphs])


# =====================================================
# GCN MODEL
# =====================================================
class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x).view(-1)  # <- final fix (sigmoid output, single probability per sample)


# =====================================================
# TRAINING + CV + TEST
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
num_node_features = train_graphs[0].x.shape[1]

os.makedirs("Results_GCN", exist_ok=True)
os.makedirs("Results_GCN/Saved_Models", exist_ok=True)

results_all_runs = []

for run in range(1, 11):
    print(f"\n================ RUN {run}/10 ================")
    torch.manual_seed(run)
    np.random.seed(run)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run)
    pr_auc_cv, roc_auc_cv, f1_cv, mcc_cv = [], [], [], []

    # ---------------- CV ----------------
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_graphs, train_labels)):
        tr_data = [train_graphs[i] for i in tr_idx]
        val_data = [train_graphs[i] for i in val_idx]

        train_loader = DataLoader(tr_data, batch_size=16, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)

        model = GCNModel(num_node_features).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=1e-4
        )

        # ---- train ----
        model.train()
        for epoch in range(50):
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.binary_cross_entropy(
                    out, data.y.view(-1).to(device)
                )
                loss.backward()
                optimizer.step()

        # ---- validation ----
        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                y_true.extend(data.y.cpu().numpy())
                y_score.extend(out.cpu().numpy())

        y_pred = [1 if p >= 0.5 else 0 for p in y_score]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        pr_auc_cv.append(pr_auc)
        roc_auc_cv.append(roc_auc)
        f1_cv.append(f1)
        mcc_cv.append(mcc)

        print(f"Fold {fold+1} PR-AUC: {pr_auc:.3f} | ROC-AUC: {roc_auc:.3f} | F1: {f1:.3f} | MCC: {mcc:.3f}")

    # ---------------- FINAL TRAINING ON ALL DATA ----------------
    # This step was missing: without it, the test section below would reuse
    # 'model' from the last CV fold (trained on only 80% of train_graphs)
    # instead of a model trained on the full training set.
    final_model = GCNModel(num_node_features).to(device)
    final_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(
        final_model.parameters(), lr=0.001, weight_decay=1e-4
    )

    final_model.train()
    for epoch in range(50):
        for data in final_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = final_model(data)
            loss = F.binary_cross_entropy(
                out, data.y.view(-1).to(device)
            )
            loss.backward()
            optimizer.step()

    # ---------------- TEST ----------------
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
    final_model.eval()  # <- uses the model trained on full train_graphs, not the last CV fold's model

    y_true, y_score = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = final_model(data)
            y_true.extend(data.y.cpu().numpy())
            y_score.extend(out.cpu().numpy())

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc_test = auc(recall, precision)

    y_pred_test = [1 if p >= 0.5 else 0 for p in y_score]
    roc_auc_test = roc_auc_score(y_true, y_score)
    f1_test = f1_score(y_true, y_pred_test)
    mcc_test = matthews_corrcoef(y_true, y_pred_test)

    # ---------------- SAVE FINAL MODEL ----------------
    # Each run trains a genuinely different final_model (see header note), so
    # a separate model is saved per run rather than only the last one.
    model_file = f"Results_GCN/Saved_Models/GCN_final_model_run{run}.pt"
    torch.save(final_model.state_dict(), model_file)

    # ---------------- TEST RESULTS (XGB-STYLE TABLE) ----------------
    results_df = pd.DataFrame({
        "Active_Prob": y_score,
        "Inactive_Prob": 1 - np.array(y_score),
        "Predicted_Class": [
            "Active" if p >= 0.5 else "Inactive" for p in y_score
        ],
        "Real_Class": [
            "Active" if y == 1 else "Inactive" for y in y_true
        ]
    })

    results_df.to_csv(
        f"Results_GCN/GCN_test_results_run{run}.csv",
        index=False
    )

    results_all_runs.append({
        "run": run,
        "cv_mean_pr_auc": np.mean(pr_auc_cv),
        "cv_std_pr_auc": np.std(pr_auc_cv),
        "cv_mean_roc_auc": np.mean(roc_auc_cv),
        "cv_std_roc_auc": np.std(roc_auc_cv),
        "cv_mean_f1": np.mean(f1_cv),
        "cv_std_f1": np.std(f1_cv),
        "cv_mean_mcc": np.mean(mcc_cv),
        "cv_std_mcc": np.std(mcc_cv),
        "test_pr_auc": pr_auc_test,
        "test_roc_auc": roc_auc_test,
        "test_f1": f1_test,
        "test_mcc": mcc_test
    })

    print(f"Test PR-AUC: {pr_auc_test:.3f} | ROC-AUC: {roc_auc_test:.3f} | F1: {f1_test:.3f} | MCC: {mcc_test:.3f}")

# =====================================================
# FINAL SUMMARY
# =====================================================
summary_df = pd.DataFrame(results_all_runs)
summary_df.to_csv("Results_GCN/GCN_summary_10runs.csv", index=False)

print("\n================ FINAL SUMMARY ================")
print(summary_df)