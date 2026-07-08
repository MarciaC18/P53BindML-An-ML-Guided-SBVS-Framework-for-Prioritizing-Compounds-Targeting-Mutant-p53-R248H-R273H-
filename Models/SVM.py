# =========================================================
# FULL SCRIPT: PLEC + CV + FINAL TRAIN + TEST (SVM)
# 10 RUNS WITH COMPLETE METRICS
# =========================================================
#
# METHODOLOGICAL NOTES
# ---------------------
#
# Source of variability across the 10 runs:
# Each run varies the random seed used for (1) the StratifiedKFold split in
# the outer cross-validation, and (2) SVC's internal probability calibration
# (Platt scaling), which itself runs a 5-fold CV when probability=True.
# The RBF-kernel SVM optimization itself (libsvm) is deterministic given
# fixed training data, so svm_final's decision boundary is consistent across
# runs; the 10 repetitions quantify variability from CV partitioning and
# probability calibration, consistent with a repeated stratified CV design.
#
# Feature representation:
# PLEC is generated with sparse=False (dense folded fingerprint). Feature
# scaling is conditional on the fingerprint's actual value range, which is
# verified empirically at runtime in process_dataset() rather than assumed
# (see the is_binary check below). If features are count-based rather than
# binary, a zero-preserving scaler (e.g. MaxAbsScaler) is recommended before
# fitting the RBF-kernel SVM, since RBF is sensitive to feature magnitude.
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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, matthews_corrcoef, auc as auc_prc
from sklearn.model_selection import StratifiedKFold

# -----------------------------
# 1. LOGGING CONFIG
# -----------------------------
logging.basicConfig(
    filename="plec_pipeline_svm.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("===== START PIPELINE: PLEC + SVM 10 RUNS =====")

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
    mol = row['mol']
    receptor_name = row['receptor']
    receptor = receptors[receptor_name]

    feature = PLEC(
        mol,
        protein=receptor,
        size=4092,
        depth_protein=4,
        depth_ligand=2,
        distance_cutoff=4.5,
        sparse=False
    )
    return feature

# -----------------------------
# 3. DATASET PROCESSING WITH VALIDATIONS
# -----------------------------
def process_dataset(folder, dataset_name="TRAIN", num_cores=16):
    logging.info(f"===== PROCESSING {dataset_name} =====")

    # CSV
    csv_file = Path(folder) / "data" / f"{dataset_name.lower()}_data.csv"
    df = pd.read_csv(csv_file, sep=';')

    # Ligands
    ligand_files = glob.glob(str(Path(folder) / "ligands" / "*.mol2"))
    ligands, names = [], []
    for f in ligand_files:
        mol = next(oddt.toolkit.readfile('mol2', f))
        ligands.append(mol)
        names.append(Path(f).stem)
    ligands_df = pd.DataFrame({'mol': ligands, 'mol_name': names})

    # Check for duplicate mol_name before merging, since a duplicate would
    # silently multiply rows in the merge below.
    dup_names = ligands_df['mol_name'][ligands_df['mol_name'].duplicated()].unique()
    if len(dup_names) > 0:
        raise RuntimeError(f"[{dataset_name}] Duplicate mol_name found in ligand files: {dup_names}")

    df = ligands_df.merge(df, on='mol_name', how='left')

    if df.isna().sum().sum() != 0:
        raise RuntimeError(f"[{dataset_name}] Merge error (NaNs detected)")

    # Receptors
    receptor_files = glob.glob(str(Path(folder) / "receptor" / "*.mol2"))
    receptors_dict = {}
    for f in receptor_files:
        r_name = Path(f).stem
        receptors_dict[r_name] = next(oddt.toolkit.readfile('mol2', f))
    missing = df['receptor'][~df['receptor'].isin(receptors_dict.keys())].unique()
    if len(missing) > 0:
        raise RuntimeError(f"[{dataset_name}] Invalid receptors: {missing}")

    # Labels — verify only the expected classes are present before anything else
    unexpected_labels = set(df['activity'].unique()) - set(Activity_Dict.keys())
    if unexpected_labels:
        raise RuntimeError(
            f"[{dataset_name}] Unexpected activity labels found: {unexpected_labels}. "
            f"Expected only {list(Activity_Dict.keys())}"
        )
    y = df['activity']
    logging.info(f"[{dataset_name}] Class counts: {y.value_counts().to_dict()}")

    # PLEC — binary vs. count nature is verified below, not assumed here
    features = Parallel(
        n_jobs=num_cores,
        backend="multiprocessing"
    )(
        delayed(plec_from_row)(row, receptors_dict)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )

    # Validations
    sums = np.array([f.sum() for f in features])
    logging.info(f"[{dataset_name}] PLEC sum min/max/mean: {sums.min()} {sums.max()} {sums.mean()}")
    active_bits = [(f > 0).sum() for f in features]
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
        logging.warning(f"[{dataset_name}] PLEC features are COUNT-based, not binary "
                         f"(max value = {unique_vals.max()}). Consider whether feature "
                         f"scaling is appropriate for the RBF kernel.")
    if len(features) != len(y):
        logging.error(f"[{dataset_name}] Features-labels misalignment")
        raise RuntimeError(f"[{dataset_name}] Features-labels misalignment")

    return df, np.array(features), y

# -----------------------------
# 4. PROCESS TRAIN AND TEST (ONCE)
# -----------------------------
train_folder = "train"
train_df, train_features, Train_Class = process_dataset(train_folder, dataset_name="TRAIN")

test_folder = "test"
test_df, test_features, Test_Class = process_dataset(test_folder, dataset_name="TEST")

# -----------------------------
# 5. 10 INDEPENDENT SVM RUNS
# -----------------------------
num_runs = 10
all_runs_results = []
output_dir = "Results_SVM"
os.makedirs(output_dir, exist_ok=True)

# Convert classes to numeric using the single source-of-truth dict
y_train = np.array([Activity_Dict[item] for item in Train_Class]).flatten()
y_test = np.array([Activity_Dict[item] for item in Test_Class]).flatten()
train_features_array = np.array(train_features)
test_features_array = np.array(test_features)

# Keep a reference to the last trained model so it can be saved after the
# loop. Model selection is not performed on the test set; selecting a model
# based on test-set performance would bias the reported test metric upward,
# since it would report the best of several draws rather than a single
# unbiased measurement.
last_run_model = None

for run in range(1, num_runs + 1):
    logging.info(f"===== RUN {run} =====")
    print(f"Starting run {run}/{num_runs}...")

    run_seed = 42 + run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # -----------------------------
    # CROSS-VALIDATION
    # -----------------------------
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=run_seed)
    roc_auc_scores_cv = []
    pr_auc_scores_cv = []
    f1_scores_cv = []
    mcc_scores_cv = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features_array, y_train)):
        X_train_fold, X_val_fold = train_features_array[train_idx], train_features_array[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        svm_fold = SVC(kernel="rbf", probability=True, random_state=run_seed)
        svm_fold.fit(X_train_fold, y_train_fold)

        # ALWAYS verify which column of predict_proba corresponds to the
        # positive class — never assume column order. classes_ is sorted,
        # but confirming explicitly avoids silent errors if that ever changes.
        assert set(svm_fold.classes_) == {0, 1}, f"Unexpected classes_ in fold {fold}: {svm_fold.classes_}"
        idx_active_fold = list(svm_fold.classes_).index(1)

        y_val_prob = svm_fold.predict_proba(X_val_fold)[:, idx_active_fold]
        y_val_pred_class = (y_val_prob >= 0.5).astype(int)

        # Metrics
        roc_auc_scores_cv.append(roc_auc_score(y_val_fold, y_val_prob))
        precision_val, recall_val, _ = precision_recall_curve(y_val_fold, y_val_prob, pos_label=1)
        pr_auc_scores_cv.append(auc_prc(recall_val, precision_val))
        f1_scores_cv.append(f1_score(y_val_fold, y_val_pred_class))
        mcc_scores_cv.append(matthews_corrcoef(y_val_fold, y_val_pred_class))

    cv_results = pd.DataFrame({
        "Fold": range(1, k_folds + 1),
        "ROC_AUC": roc_auc_scores_cv,
        "PR_AUC": pr_auc_scores_cv,
        "F1": f1_scores_cv,
        "MCC": mcc_scores_cv
    })
    cv_results.loc[len(cv_results)] = ["Mean", np.mean(roc_auc_scores_cv), np.mean(pr_auc_scores_cv),
                                        np.mean(f1_scores_cv), np.mean(mcc_scores_cv)]
    cv_results.loc[len(cv_results)] = ["Std", np.std(roc_auc_scores_cv), np.std(pr_auc_scores_cv),
                                        np.std(f1_scores_cv), np.std(mcc_scores_cv)]
    cv_filename = os.path.join(output_dir, f"SVM_CV_run{run}_{timestamp}.csv")
    cv_results.to_csv(cv_filename, index=False)

    # -----------------------------
    # FINAL TRAINING
    # -----------------------------
    svm_final = SVC(kernel="rbf", probability=True, random_state=run_seed)
    svm_final.fit(train_features_array, y_train)

    # ALWAYS verify class order before indexing predict_proba — do not assume
    # column 1 = Active / column 0 = Inactive. classes_ is the ground truth.
    assert set(svm_final.classes_) == {0, 1}, f"Unexpected classes_ in run {run}: {svm_final.classes_}"
    idx_active = list(svm_final.classes_).index(1)
    idx_inactive = list(svm_final.classes_).index(0)
    logging.info(f"[Run {run}] svm_final.classes_={list(svm_final.classes_)} -> "
                 f"idx_active={idx_active} idx_inactive={idx_inactive}")

    # TRAIN metrics
    train_prob = svm_final.predict_proba(train_features_array)[:, idx_active]
    train_class_pred = (train_prob >= 0.5).astype(int)
    train_acc = accuracy_score(y_train, train_class_pred)
    train_f1 = f1_score(y_train, train_class_pred)
    train_mcc = matthews_corrcoef(y_train, train_class_pred)
    precision_train, recall_train, _ = precision_recall_curve(y_train, train_prob, pos_label=1)
    train_pr_auc = auc_prc(recall_train, precision_train)
    train_auc_roc = roc_auc_score(y_train, train_prob)

    # -----------------------------
    # TEST
    # -----------------------------
    test_prob = svm_final.predict_proba(test_features_array)[:, idx_active]
    test_class_pred = (test_prob >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_class_pred)
    test_f1 = f1_score(y_test, test_class_pred)
    test_mcc = matthews_corrcoef(y_test, test_class_pred)
    precision_test, recall_test, _ = precision_recall_curve(y_test, test_prob, pos_label=1)
    test_pr_auc = auc_prc(recall_test, precision_test)
    test_auc_roc = roc_auc_score(y_test, test_prob)

    # Build results table with numeric 0/1 first, then convert to readable
    # class labels using the single Activity_Dict mapping (inverse), so the
    # CSV always shows "Active"/"Inactive" consistently with how classes
    # were verified above.
    plec_result_svm = pd.DataFrame({
        "Active_Prob": test_prob,
        "Inactive_Prob": svm_final.predict_proba(test_features_array)[:, idx_inactive],
        "Predicted_Class": test_class_pred,
        "Real_Class": y_test
    })
    plec_result_svm.replace({1: POSITIVE_CLASS, 0: NEGATIVE_CLASS}, inplace=True)

    csv_filename = os.path.join(output_dir, f"SVM_results_run{run}_{timestamp}.csv")
    log_filename = os.path.join(output_dir, f"SVM_log_run{run}_{timestamp}.txt")
    plec_result_svm.to_csv(csv_filename, index=False)

    # Keep this run's model as the current "last run" (overwritten each
    # iteration, so after the loop it holds the model from run == num_runs)
    last_run_model = svm_final

    # -----------------------------
    # STORE RESULTS
    # -----------------------------
    all_runs_results.append({
        "run": run,
        "cv_mean_roc_auc": np.mean(roc_auc_scores_cv),
        "cv_std_roc_auc": np.std(roc_auc_scores_cv),
        "cv_mean_pr_auc": np.mean(pr_auc_scores_cv),
        "cv_mean_f1": np.mean(f1_scores_cv),
        "cv_mean_mcc": np.mean(mcc_scores_cv),
        "train_acc": train_acc,
        "train_roc_auc": train_auc_roc,
        "train_pr_auc": train_pr_auc,
        "train_f1": train_f1,
        "train_mcc": train_mcc,
        "test_acc": test_acc,
        "test_roc_auc": test_auc_roc,
        "test_pr_auc": test_pr_auc,
        "test_f1": test_f1,
        "test_mcc": test_mcc
    })

    with open(log_filename, "w") as f:
        f.write(f"=== Run {run} ===\n\n")
        f.write(cv_results.to_string(index=False))
        f.write(f"\nTrain Acc: {train_acc:.3f} | ROC-AUC: {train_auc_roc:.3f} | PR-AUC: {train_pr_auc:.3f}\n")
        f.write(f"Test Acc: {test_acc:.3f} | ROC-AUC: {test_auc_roc:.3f} | PR-AUC: {test_pr_auc:.3f}\n")
        f.write(f"First rows of predictions:\n{plec_result_svm.head().to_string()}\n")

# -----------------------------
# 6. FINAL SUMMARY
# -----------------------------
summary_df = pd.DataFrame(all_runs_results)
summary_csv = os.path.join(output_dir, f"summary_all_runs_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
summary_df.to_csv(summary_csv, index=False)

print("===== FINAL SUMMARY =====")
print(summary_df.describe())

# -----------------------------
# 7. SAVE FINAL MODEL AND PLEC PARAMETERS FOR VS
# -----------------------------
# The model saved here corresponds to the last run (run == num_runs). Model
# selection is not performed on the test set, to keep the reported test
# metrics an unbiased estimate of generalization performance (see repeated-CV
# note above). As noted, the RBF-kernel SVM decision boundary is consistent
# across runs given fixed training data, so this corresponds to a
# representative final model rather than a specifically "selected" one.
model_dir = os.path.join(output_dir, "Saved_Models")
os.makedirs(model_dir, exist_ok=True)

final_model_file = os.path.join(model_dir, f"SVM_final_model_run{num_runs}.pkl")
dump(last_run_model, final_model_file)
print(f"✅ Final SVM model saved (run {num_runs}) in: {final_model_file}")

# Save PLEC parameters for reproducibility
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

logging.info("===== END OF PIPELINE: PLEC + SVM 10 RUNS =====")