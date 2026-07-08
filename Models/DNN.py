# =========================================================
# FULL SCRIPT: PLEC + CV + FINAL TRAIN + TEST (DNN)
# 10 RUNS WITH COMPLETE METRICS
# =========================================================
#
# NOTES
# ---------------------
#
# Source of variability across the 10 runs:
# DNN has genuine per-run stochasticity:
# weight initialization, Dropout masking, and mini-batch shuffling are all
# governed by TensorFlow's/NumPy's global random state, reseeded per run
# (and per fold, offset by fold index) via tf.random.set_seed/np.random.seed.
# Each run therefore trains a genuinely different set of weights, not just a
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
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
    filename="plec_pipeline_dnn_10runs.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("===== START PIPELINE: PLEC + DNN 10 RUNS =====")

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
    return PLEC(
        row["mol"],
        protein=receptors[row["receptor"]],
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
    csv_file = Path(folder) / "data" / f"{dataset_name.lower()}_data.csv"
    df = pd.read_csv(csv_file, sep=";")

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

    receptor_files = glob.glob(str(Path(folder) / "receptor" / "*.mol2"))
    receptors = {Path(f).stem: next(oddt.toolkit.readfile("mol2", f)) for f in receptor_files}

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
        logging.warning(f"[{dataset_name}] PLEC features are count-based (max value = "
                         f"{unique_vals.max()}). Unlike tree-based models, this DNN's "
                         f"gradient-based optimization may benefit from feature scaling; "
                         f"consider adding one if this is observed.")

    if len(features) != len(y):
        raise RuntimeError(f"[{dataset_name}] Features/labels misalignment")

    return np.array(features), y

# -----------------------------
# 4. LOAD TRAIN / TEST
# -----------------------------
train_features, Train_Class = process_dataset("train", "TRAIN")
test_features, Test_Class = process_dataset("test", "TEST")

y_train = np.array([Activity_Dict[i] for i in Train_Class]).astype("float32")
y_test = np.array([Activity_Dict[i] for i in Test_Class]).astype("float32")

# -----------------------------
# 5. DNN
# -----------------------------
def create_dnn():
    model = keras.Sequential([
        layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# 6. 10 RUNS
# -----------------------------
num_runs = 10
output_dir = "Results_DNN"
os.makedirs(output_dir, exist_ok=True)
all_runs = []

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

    tf.random.set_seed(run_seed)
    np.random.seed(run_seed)

    # -------- CROSS-VALIDATION --------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
    roc_auc_cv, pr_auc_cv, f1_cv, mcc_cv = [], [], [], []

    for fold, (tr, val) in enumerate(skf.split(train_features, y_train), 1):
        tf.random.set_seed(run_seed + fold)

        model = create_dnn()
        model.fit(train_features[tr], y_train[tr], epochs=100, batch_size=64, verbose=0)

        val_prob = model.predict(train_features[val])[:, 0]
        val_pred = (val_prob >= 0.5).astype(int)

        roc_auc_cv.append(roc_auc_score(y_train[val], val_prob))
        p, r, _ = precision_recall_curve(y_train[val], val_prob, pos_label=1)
        pr_auc_cv.append(auc_prc(r, p))
        f1_cv.append(f1_score(y_train[val], val_pred))
        mcc_cv.append(matthews_corrcoef(y_train[val], val_pred))

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
    cv_df.to_csv(os.path.join(output_dir, f"DNN_CV_run{run}_{timestamp}.csv"), index=False)

    # -------- FINAL TRAINING --------
    final_model = create_dnn()
    final_model.fit(train_features, y_train, epochs=100, batch_size=500, verbose=0)

    # Sanity check on model output shape/range before indexing — this model
    # has a single sigmoid output (no classes_ attribute like sklearn), so
    # column 0 is always the positive-class ("Active") probability by
    # construction of create_dnn(); verify that assumption holds at runtime.
    train_prob = final_model.predict(train_features)[:, 0]
    assert train_prob.ndim == 1 and train_prob.min() >= 0.0 and train_prob.max() <= 1.0, \
        "Unexpected DNN output shape/range — expected a single sigmoid probability per sample"
    train_pred = (train_prob >= 0.5).astype(int)

    train_acc = accuracy_score(y_train, train_pred)
    train_roc_auc = roc_auc_score(y_train, train_prob)
    p, r, _ = precision_recall_curve(y_train, train_prob, pos_label=1)
    train_pr_auc = auc_prc(r, p)
    train_f1 = f1_score(y_train, train_pred)
    train_mcc = matthews_corrcoef(y_train, train_pred)

    # -------- TEST --------
    test_prob = final_model.predict(test_features)[:, 0]
    test_pred = (test_prob >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_roc_auc = roc_auc_score(y_test, test_prob)
    p, r, _ = precision_recall_curve(y_test, test_prob, pos_label=1)
    test_pr_auc = auc_prc(r, p)
    test_f1 = f1_score(y_test, test_pred)
    test_mcc = matthews_corrcoef(y_test, test_pred)

    results_df = pd.DataFrame({
        "Active_Prob": test_prob,
        "Predicted_Class": test_pred,
        "Real_Class": y_test
    })
    results_df.replace({1: POSITIVE_CLASS, 0: NEGATIVE_CLASS}, inplace=True)
    results_df.to_csv(os.path.join(output_dir, f"DNN_results_run{run}_{timestamp}.csv"), index=False)

    # Keep this run's model as the current "last run" (overwritten each
    # iteration, so after the loop it holds the model from run == num_runs)
    last_run_model = final_model

    all_runs.append({
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
# 7. FINAL SUMMARY
# -----------------------------
summary = pd.DataFrame(all_runs)
summary.to_csv(os.path.join(output_dir, "DNN_summary_10runs.csv"), index=False)

print(summary)

# -----------------------------
# 8. SAVE FINAL MODEL AND PLEC PARAMETERS FOR VS
# -----------------------------
# The model saved here corresponds to the last run (run == num_runs). Model
# selection is not performed on the test set, to keep the reported test
# metrics an unbiased estimate of generalization performance.
model_dir = os.path.join(output_dir, "Saved_Models")
os.makedirs(model_dir, exist_ok=True)

final_model_file = os.path.join(model_dir, f"DNN_final_model_run{num_runs}.keras")
last_run_model.save(final_model_file)
print(f"✅ Final DNN model saved (run {num_runs}) in: {final_model_file}")

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

logging.info("===== END PIPELINE: PLEC + DNN 10 RUNS =====")
