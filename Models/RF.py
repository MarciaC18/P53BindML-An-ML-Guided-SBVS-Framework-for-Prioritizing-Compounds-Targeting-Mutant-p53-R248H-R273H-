# =========================================================
# FULL SCRIPT: PLEC + CV + FINAL TRAIN + TEST (RF)
# 10 RUNS WITH COMPLETE METRICS
# =========================================================
#
# NOTES
# ---------------------
#
# Source of variability across the 10 runs:
# Unlike an RBF-kernel SVM, RandomForestClassifier has genuine per-run
# stochasticity: with bootstrap=True (default), each tree is trained on a
# bootstrap resample of the training data, and at each split only a random
# subset of features is considered (max_features="sqrt"). random_state
# controls both of these, so different seeds produce genuinely different
# forests, not just different CV splits or calibration noise. The outer
# StratifiedKFold split is also reseeded per run.
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
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
    auc as auc_prc
)
from sklearn.model_selection import StratifiedKFold
 
# -----------------------------
# 1. LOGGING CONFIG
# -----------------------------
logging.basicConfig(
    filename="plec_pipeline_RF.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("===== START PIPELINE: PLEC + RF (10 RUNS) =====")
 
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
 
    ligand_files = glob.glob(str(Path(folder) / "ligands" / "*.mol2"))
    ligands, names = [], []
    for f in ligand_files:
        mol = next(oddt.toolkit.readfile("mol2", f))
        ligands.append(mol)
        names.append(Path(f).stem)
 
    ligands_df = pd.DataFrame({"mol": ligands, "mol_name": names})
 
    # Check for duplicate mol_name before merging, since a duplicate would
    # silently multiply rows in the merge below.
    dup_names = ligands_df["mol_name"][ligands_df["mol_name"].duplicated()].unique()
    if len(dup_names) > 0:
        raise RuntimeError(f"[{dataset_name}] Duplicate mol_name found in ligand files: {dup_names}")
 
    df = ligands_df.merge(df, on="mol_name", how="left")
 
    if df.isna().sum().sum() != 0:
        raise RuntimeError(f"[{dataset_name}] Merge error (NaNs detected)")
 
    receptor_files = glob.glob(str(Path(folder) / "receptor" / "*.mol2"))
    receptors = {Path(f).stem: next(oddt.toolkit.readfile("mol2", f)) for f in receptor_files}
 
    missing = df["receptor"][~df["receptor"].isin(receptors)].unique()
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
 
    features = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(plec_from_row)(row, receptors) for _, row in tqdm(df.iterrows(), total=len(df))
    )
 
    # Explicitly verify whether PLEC output is truly binary (0/1) or count-based.
    # PLEC uses hashed folding (sparse=False), so bit collisions can produce
    # values > 1 even though it's often loosely described as "binary". Never
    # assume this — check it directly on the actual computed features.
    unique_vals = np.unique(np.concatenate(features))
    is_binary = set(unique_vals.astype(int)) <= {0, 1}
    logging.info(f"[{dataset_name}] PLEC unique values sample: {unique_vals[:10]} ... "
                 f"max={unique_vals.max()} | is_binary={is_binary}")
    if not is_binary:
        logging.info(f"[{dataset_name}] PLEC features are count-based (max value = "
                      f"{unique_vals.max()}). No action needed for Random Forest, since "
                      f"tree-based models are invariant to feature scaling.")
 
    if len(features) != len(y):
        raise RuntimeError(f"[{dataset_name}] Features-labels misalignment")
 
    return df, np.array(features), y
 
# -----------------------------
# 4. PROCESS TRAIN AND TEST
# -----------------------------
train_df, train_features, Train_Class = process_dataset("train", "TRAIN")
test_df, test_features, Test_Class = process_dataset("test", "TEST")
 
y_train = np.array([Activity_Dict[i] for i in Train_Class])
y_test = np.array([Activity_Dict[i] for i in Test_Class])
 
train_features = np.array(train_features)
test_features = np.array(test_features)
 
num_runs = 10
output_dir = "Results_RF"
os.makedirs(output_dir, exist_ok=True)
 
all_runs_results = []
 
# Keep a reference to the last trained model so it can be saved after the
# loop. Model selection is not performed on the test set; selecting a model
# based on test-set performance would bias the reported test metric upward,
# since it would report the best of several draws rather than a single
# unbiased measurement.
last_run_model = None
 
# -----------------------------
# 5. 10 RF RUNS
# -----------------------------
for run in range(1, num_runs + 1):
    logging.info(f"===== RUN {run} =====")
    print(f"Starting run {run}/{num_runs}")
 
    seed = 42 + run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
 
    roc_auc_cv, pr_auc_cv, f1_cv, mcc_cv = [], [], [], []
 
    # ---------- CROSS-VALIDATION ----------
    for fold, (tr, val) in enumerate(skf.split(train_features, y_train), 1):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            max_features="sqrt",
            n_jobs=30,
            random_state=seed
        )
        rf.fit(train_features[tr], y_train[tr])
 
        # ALWAYS verify which column of predict_proba corresponds to the
        # positive class — never assume column order.
        assert set(rf.classes_) == {0, 1}, f"Unexpected classes_ in fold {fold}: {rf.classes_}"
        class_to_index = {cls: i for i, cls in enumerate(rf.classes_)}
        idx_active = class_to_index[1]
 
        prob = rf.predict_proba(train_features[val])[:, idx_active]
        pred = (prob >= 0.5).astype(int)
 
        roc_auc_cv.append(roc_auc_score(y_train[val], prob))
        p, r, _ = precision_recall_curve(y_train[val], prob, pos_label=1)
        pr_auc_cv.append(auc_prc(r, p))
        f1_cv.append(f1_score(y_train[val], pred))
        mcc_cv.append(matthews_corrcoef(y_train[val], pred))
 
    # Build CV summary explicitly with string labels for Mean/Std rows,
    # instead of cv_df.mean()/.std(), which would incorrectly average the
    # numeric "Fold" column as well and overwrite the Fold label.
    cv_df = pd.DataFrame({
        "Fold": range(1, 6),
        "ROC_AUC": roc_auc_cv,
        "PR_AUC": pr_auc_cv,
        "F1": f1_cv,
        "MCC": mcc_cv
    })
    cv_df.loc[len(cv_df)] = ["Mean", np.mean(roc_auc_cv), np.mean(pr_auc_cv),
                              np.mean(f1_cv), np.mean(mcc_cv)]
    cv_df.loc[len(cv_df)] = ["Std", np.std(roc_auc_cv), np.std(pr_auc_cv),
                              np.std(f1_cv), np.std(mcc_cv)]
    cv_df.to_csv(f"{output_dir}/RF_CV_run{run}_{timestamp}.csv", index=False)
 
    # ---------- FINAL TRAINING ----------
    rf_final = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        max_features="sqrt",
        n_jobs=30,
        random_state=seed
    )
    rf_final.fit(train_features, y_train)
 
    # ALWAYS verify class order before indexing predict_proba — do not assume
    # column 1 = Active / column 0 = Inactive. classes_ is the ground truth.
    assert set(rf_final.classes_) == {0, 1}, f"Unexpected classes_ in run {run}: {rf_final.classes_}"
    class_to_index = {cls: i for i, cls in enumerate(rf_final.classes_)}
    idx_active = class_to_index[1]
    idx_inactive = class_to_index[0]
    logging.info(f"[Run {run}] rf_final.classes_={list(rf_final.classes_)} -> "
                 f"idx_active={idx_active} idx_inactive={idx_inactive}")
 
    train_prob = rf_final.predict_proba(train_features)[:, idx_active]
    train_pred = (train_prob >= 0.5).astype(int)
 
    train_acc = accuracy_score(y_train, train_pred)
    train_roc_auc = roc_auc_score(y_train, train_prob)
    p, r, _ = precision_recall_curve(y_train, train_prob, pos_label=1)
    train_pr_auc = auc_prc(r, p)
    train_f1 = f1_score(y_train, train_pred)
    train_mcc = matthews_corrcoef(y_train, train_pred)
 
    # ---------- TEST ----------
    test_prob = rf_final.predict_proba(test_features)[:, idx_active]
    test_pred = (test_prob >= 0.5).astype(int)
 
    test_acc = accuracy_score(y_test, test_pred)
    test_roc_auc = roc_auc_score(y_test, test_prob)
    p, r, _ = precision_recall_curve(y_test, test_prob, pos_label=1)
    test_pr_auc = auc_prc(r, p)
    test_f1 = f1_score(y_test, test_pred)
    test_mcc = matthews_corrcoef(y_test, test_pred)
 
    results_df = pd.DataFrame({
        "Active_Prob": test_prob,
        "Inactive_Prob": rf_final.predict_proba(test_features)[:, idx_inactive],
        "Predicted_Class": test_pred,
        "Real_Class": y_test
    })
    results_df.replace({1: POSITIVE_CLASS, 0: NEGATIVE_CLASS}, inplace=True)
    results_df.to_csv(f"{output_dir}/RF_results_run{run}_{timestamp}.csv", index=False)
 
    # Keep this run's model as the current "last run" (overwritten each
    # iteration, so after the loop it holds the model from run == num_runs)
    last_run_model = rf_final
 
    all_runs_results.append({
        "run": run,
        "cv_mean_roc_auc": np.mean(roc_auc_cv),
        "cv_std_roc_auc": np.std(roc_auc_cv),
        "cv_mean_pr_auc": np.mean(pr_auc_cv),
        "cv_mean_f1": np.mean(f1_cv),
        "cv_mean_mcc": np.mean(mcc_cv),
        "train_acc": train_acc,
        "train_roc_auc": train_roc_auc,
        "train_pr_auc": train_pr_auc,
        "train_f1": train_f1,
        "train_mcc": train_mcc,
        "test_acc": test_acc,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_f1": test_f1,
        "test_mcc": test_mcc
    })
 
# -----------------------------
# 6. FINAL SUMMARY
# -----------------------------
summary_df = pd.DataFrame(all_runs_results)
summary_df.to_csv(
    f"{output_dir}/summary_all_runs_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
    index=False
)
 
print(summary_df)
 
# -----------------------------
# 7. SAVE FINAL MODEL AND PLEC PARAMETERS FOR VS
# -----------------------------
# The model saved here corresponds to the last run (run == num_runs).
model_dir = os.path.join(output_dir, "Saved_Models")
os.makedirs(model_dir, exist_ok=True)
 
final_model_file = os.path.join(model_dir, f"RF_final_model_run{num_runs}.pkl")
dump(last_run_model, final_model_file)
print(f"✅ Final RF model saved (run {num_runs}) in: {final_model_file}")
 
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
 
logging.info("===== END PIPELINE: PLEC + RF (10 RUNS) =====")
