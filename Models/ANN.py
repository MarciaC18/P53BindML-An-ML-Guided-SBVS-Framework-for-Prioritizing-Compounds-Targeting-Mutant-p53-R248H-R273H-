# =========================================================
# FULL SCRIPT: PLEC + CV + FINAL TRAIN + TEST (ANN)
# 10 RUNS WITH COMPLETE METRICS
# =========================================================
#
# NOTES
# ---------------------
#
# Source of variability across the 10 runs:
# Unlike an RBF-kernel SVM, MLPClassifier has genuine per-run stochasticity:
# random_state controls the initial weight values and, since the default
# solver is "adam" (a stochastic gradient-based optimizer) with shuffle=True
# by default, also controls the sample shuffling between iterations. Each
# run therefore trains a genuinely different set of weights, not just a
# different CV split. The outer StratifiedKFold split is also reseeded per
# run.
#
# Feature representation:
# PLEC is generated with sparse=False (dense folded fingerprint). Whether the
# resulting values are binary or count-based is verified empirically at
# runtime in process_dataset() rather than assumed (see the is_binary check
# below).
# =========================================================

# -----------------------------
# 0. IMPORTS
# -----------------------------
import pandas as pd
import numpy as np
import oddt
from oddt.fingerprints import PLEC
from joblib import Parallel, delayed, dump
from tqdm import tqdm
import logging
from pathlib import Path
import glob
import os
import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
    auc as auc_prc
)

# -----------------------------
# 1. LOGGING CONFIG
# -----------------------------
logging.basicConfig(
    filename="plec_pipeline_ann_10runs.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("===== START PIPELINE: PLEC + ANN 10 RUNS =====")

# -----------------------------
# CLASS LABEL CONSTANTS
# -----------------------------
# Keep label mapping in one place so it's unambiguous everywhere in the script.
POSITIVE_CLASS = "Active"
NEGATIVE_CLASS = "Inactive"
Activity_Dict = {POSITIVE_CLASS: 1, NEGATIVE_CLASS: 0}

# -----------------------------
# 2. GLOBAL FUNCTION TO COMPUTE PLEC
# -----------------------------
def plec_from_row(row, receptors):
    mol = row["mol"]
    receptor = receptors[row["receptor"]]

    return PLEC(
        mol,
        protein=receptor,
        size=4092,
        depth_protein=4,
        depth_ligand=2,
        distance_cutoff=4.5,
        sparse=False
    )

# -----------------------------
# 3. DATASET PROCESSING WITH VALIDATIONS
# -----------------------------
def process_dataset(folder, dataset_name="TRAIN", num_cores=16):
    logging.info(f"===== PROCESSING {dataset_name} =====")

    csv_file = Path(folder) / "data" / f"{dataset_name.lower()}_data.csv"
    df = pd.read_csv(csv_file, sep=";")

    # Ligands
    ligand_files = glob.glob(str(Path(folder) / "ligands" / "*.mol2"))
    ligands, names = [], []
    for f in ligand_files:
        ligands.append(next(oddt.toolkit.readfile("mol2", f)))
        names.append(Path(f).stem)

    lig_df = pd.DataFrame({"mol": ligands, "mol_name": names})

    # Check for duplicate mol_name before merging, since a duplicate would
    # silently multiply rows in the merge below.
    dup_names = lig_df["mol_name"][lig_df["mol_name"].duplicated()].unique()
    if len(dup_names) > 0:
        raise RuntimeError(f"[{dataset_name}] Duplicate mol_name found in ligand files: {dup_names}")

    df = lig_df.merge(df, on="mol_name", how="left")
    if df.isna().sum().sum() != 0:
        raise RuntimeError(f"[{dataset_name}] NaNs detected")

    # Receptors
    receptor_files = glob.glob(str(Path(folder) / "receptor" / "*.mol2"))
    receptors = {Path(f).stem: next(oddt.toolkit.readfile("mol2", f)) for f in receptor_files}

    missing = df["receptor"][~df["receptor"].isin(receptors.keys())].unique()
    if len(missing) > 0:
        raise RuntimeError(f"[{dataset_name}] Invalid receptors: {missing}")

    # Labels — verify only the expected classes are present before anything else
    unexpected_labels = set(df["activity"].unique()) - set(Activity_Dict.keys())
    if unexpected_labels:
        raise RuntimeError(
            f"[{dataset_name}] Unexpected activity labels found: {unexpected_labels}. "
            f"Expected only {list(Activity_Dict.keys())}"
        )
    y = df["activity"]
    logging.info(f"[{dataset_name}] Class counts: {y.value_counts().to_dict()}")

    # -------- PLEC --------
    logging.info(f"[{dataset_name}] Starting PLEC computation")
    features = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(plec_from_row)(row, receptors) for _, row in tqdm(df.iterrows(), total=len(df))
    )
    logging.info(f"[{dataset_name}] Finished PLEC computation")

    # -------- VALIDATIONS --------
    sums = np.array([f.sum() for f in features])
    active_bits = [(f > 0).sum() for f in features]
    logging.info(f"[{dataset_name}] PLEC sum min/max/mean: {sums.min()} {sums.max()} {sums.mean():.2f}")
    logging.info(f"[{dataset_name}] Active bits min/max: {min(active_bits)} {max(active_bits)}")

    # Explicitly verify whether PLEC output is truly binary (0/1) or count-based.
    # PLEC uses hashed folding (sparse=False), so bit collisions can produce
    # values > 1 even though it's often loosely described as "binary". Never
    # assume this — check it directly on the actual computed features.
    unique_vals = np.unique(np.concatenate(features))
    is_binary = set(unique_vals.astype(int)) <= {0, 1}
    logging.info(f"[{dataset_name}] PLEC unique values sample: {unique_vals[:10]} ... "
                 f"max={unique_vals.max()} | is_binary={is_binary}")
    if not is_binary:
        logging.warning(f"[{dataset_name}] PLEC features are count-based (max value = "
                         f"{unique_vals.max()}). Unlike tree-based models, this MLP's "
                         f"gradient-based optimization may benefit from feature scaling; "
                         f"consider adding one if this is observed.")

    if len(features) != len(y):
        raise RuntimeError(f"[{dataset_name}] Features/labels misalignment")

    return df, np.array(features), y

# -----------------------------
# 4. LOAD TRAIN / TEST
# -----------------------------
train_df, train_features, Train_Class = process_dataset("train", "TRAIN")
test_df, test_features, Test_Class = process_dataset("test", "TEST")

train_features = np.array(train_features)
test_features = np.array(test_features)

# -----------------------------
# 5. 10 INDEPENDENT ANN RUNS
# -----------------------------
num_runs = 10
output_dir = "Results_ANN"
os.makedirs(output_dir, exist_ok=True)
all_runs_results = []

y_train = np.array([Activity_Dict[i] for i in Train_Class])
y_test = np.array([Activity_Dict[i] for i in Test_Class])

# Keep a reference to the last trained model so it can be saved after the
# loop. Model selection is not performed on the test set; selecting a model
# based on test-set performance would bias the reported test metric upward,
# since it would report the best of several draws rather than a single
# unbiased measurement.
last_run_model = None

for run in range(1, num_runs + 1):
    run_seed = 42 + run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"===== RUN {run} | seed={run_seed} =====")
    print(f"RUN {run}/{num_runs}")

    # -----------------------------
    # CROSS-VALIDATION
    # -----------------------------
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=run_seed)

    aucs_cv, pr_aucs_cv, f1_cv, mcc_cv = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_features, y_train)):
        X_tr, X_val = train_features[tr_idx], train_features[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        ann_fold = MLPClassifier(max_iter=9000, random_state=run_seed)
        ann_fold.fit(X_tr, y_tr)

        # ALWAYS verify which column of predict_proba corresponds to the
        # positive class — never assume column order.
        assert set(ann_fold.classes_) == {0, 1}, f"Unexpected classes_ in fold {fold}: {ann_fold.classes_}"
        class_to_index = {cls: i for i, cls in enumerate(ann_fold.classes_)}
        idx_active = class_to_index[1]

        y_val_prob = ann_fold.predict_proba(X_val)[:, idx_active]
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_val_prob)
        precision, recall, _ = precision_recall_curve(y_val, y_val_prob, pos_label=1)
        pr_auc = auc_prc(recall, precision)
        f1 = f1_score(y_val, y_val_pred)
        mcc = matthews_corrcoef(y_val, y_val_pred)

        aucs_cv.append(auc)
        pr_aucs_cv.append(pr_auc)
        f1_cv.append(f1)
        mcc_cv.append(mcc)

        logging.info(f"[RUN {run}] Fold {fold+1} AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

    # Build CV summary explicitly with string labels for Mean/Std rows,
    # instead of cv_df.mean()/.std(), which would incorrectly average the
    # numeric "Fold" column as well and overwrite the Fold label.
    cv_df = pd.DataFrame({
        "Fold": range(1, k_folds + 1),
        "AUC": aucs_cv,
        "PR_AUC": pr_aucs_cv,
        "F1": f1_cv,
        "MCC": mcc_cv
    })
    cv_df.loc[len(cv_df)] = ["Mean", np.mean(aucs_cv), np.mean(pr_aucs_cv),
                              np.mean(f1_cv), np.mean(mcc_cv)]
    cv_df.loc[len(cv_df)] = ["Std", np.std(aucs_cv), np.std(pr_aucs_cv),
                              np.std(f1_cv), np.std(mcc_cv)]
    cv_df.to_csv(os.path.join(output_dir, f"ANN_CV_run{run}_{timestamp}.csv"), index=False)

    # -----------------------------
    # FINAL ANN TRAINING
    # -----------------------------
    ann_final = MLPClassifier(max_iter=9000, random_state=run_seed)
    ann_final.fit(train_features, y_train)

    # ALWAYS verify class order before indexing predict_proba — do not assume
    # column 1 = Active / column 0 = Inactive. classes_ is the ground truth.
    assert set(ann_final.classes_) == {0, 1}, f"Unexpected classes_ in run {run}: {ann_final.classes_}"
    class_to_index = {cls: i for i, cls in enumerate(ann_final.classes_)}
    idx_active = class_to_index[1]
    idx_inactive = class_to_index[0]
    logging.info(f"[Run {run}] ann_final.classes_={list(ann_final.classes_)} -> "
                 f"idx_active={idx_active} idx_inactive={idx_inactive}")

    test_prob = ann_final.predict_proba(test_features)[:, idx_active]
    test_pred = (test_prob >= 0.5).astype(int)

    precision_test, recall_test, _ = precision_recall_curve(y_test, test_prob, pos_label=1)
    test_pr_auc = auc_prc(recall_test, precision_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_mcc = matthews_corrcoef(y_test, test_pred)

    results_df = pd.DataFrame({
        "Active_Prob": test_prob,
        "Inactive_Prob": ann_final.predict_proba(test_features)[:, idx_inactive],
        "Predicted_Class": ["Active" if x == 1 else "Inactive" for x in test_pred],
        "Real_Class": Test_Class
    })
    results_df.to_csv(os.path.join(output_dir, f"ANN_results_run{run}_{timestamp}.csv"), index=False)

    # Keep this run's model as the current "last run" (overwritten each
    # iteration, so after the loop it holds the model from run == num_runs)
    last_run_model = ann_final

    # -----------------------------
    # STORE RUN METRICS
    # -----------------------------
    all_runs_results.append({
        "run": run,
        "seed": run_seed,
        "cv_mean_auc": np.mean(aucs_cv),
        "cv_std_auc": np.std(aucs_cv),
        "cv_mean_pr_auc": np.mean(pr_aucs_cv),
        "cv_mean_f1": np.mean(f1_cv),
        "cv_mean_mcc": np.mean(mcc_cv),
        "test_acc": test_acc,
        "test_pr_auc": test_pr_auc,
        "test_f1": test_f1,
        "test_mcc": test_mcc
    })

# -----------------------------
# 6. FINAL SUMMARY
# -----------------------------
summary_df = pd.DataFrame(all_runs_results)
summary_df.to_csv(
    os.path.join(output_dir, f"summary_10runs_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"),
    index=False
)

print(summary_df)

# -----------------------------
# 7. SAVE FINAL MODEL AND PLEC PARAMETERS FOR VS
# -----------------------------
# The model saved here corresponds to the last run (run == num_runs). Model
# selection is not performed on the test set, to keep the reported test
# metrics an unbiased estimate of generalization performance. 
model_dir = os.path.join(output_dir, "Saved_Models")
os.makedirs(model_dir, exist_ok=True)

final_model_file = os.path.join(model_dir, f"ANN_final_model_run{num_runs}.pkl")
dump(last_run_model, final_model_file)
print(f"✅ Final ANN model saved (run {num_runs}) in: {final_model_file}")

plec_params = {
    "size": 4092,
    "depth_protein": 4,
    "depth_ligand": 2,
    "distance_cutoff": 4.5,
    "sparse": False
}
plec_params_file = os.path.join(model_dir, "PLEC_params.pkl")
dump(plec_params, plec_params_file)
print(f"✅ PLEC parameters saved in: {plec_params_file}")

logging.info("===== END PIPELINE: PLEC + ANN 10 RUNS =====")
