import os
import glob
import random
import sys
from pathlib import Path

# Make the EchoFocus repository root importable.
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import CustomTransformer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path.home() / "echofocus" / "ventricle_project"

DATASET_CSV = BASE_DIR / "splits" / "rvlv_dataset.csv"
EMBEDDING_DIR = BASE_DIR / "embeddings" / "dicom"

CHECKPOINT = Path.home() / "echofocus" / "trained_models" / "EchoFocus_CHD" / "best_checkpoint.pt"

RESULT_DIR = BASE_DIR / "results" / "linear_probe_rvlv"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = RESULT_DIR / "frozen_features.pt"
BEST_MODEL_FILE = RESULT_DIR / "best_linear_probe.pt"
METRICS_FILE = RESULT_DIR / "metrics.csv"
PREDICTIONS_FILE = RESULT_DIR / "predictions.csv"


SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("EchoFocus — RV vs LV Linear Probe")
print("=" * 70)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset:")
print(DATASET_CSV)

df = pd.read_csv(DATASET_CSV)

print("\nColumns:")
print(df.columns.tolist())

print("\nRows:", len(df))


# ------------------------------------------------------------
# Identify required columns
# ------------------------------------------------------------

def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


PATIENT_COL = find_column([
    "patient_id",
    "PatientID",
    "patient",
])

FOLDER_COL = find_column([
    "folder",
    "Folder",
])

LABEL_COL = find_column([
    "label",
    "Label",
    "target",
    "Target",
])

SPLIT_COL = find_column([
    "split",
    "Split",
    "dataset_split",
])


print("\nDetected columns:")
print("patient:", PATIENT_COL)
print("folder :", FOLDER_COL)
print("label  :", LABEL_COL)
print("split  :", SPLIT_COL)


if PATIENT_COL is None:
    raise ValueError("Could not find patient_id column.")

if FOLDER_COL is None:
    raise ValueError("Could not find folder column.")

if LABEL_COL is None:
    raise ValueError("Could not find label column.")

if SPLIT_COL is None:
    raise ValueError("Could not find split column.")


# Convert IDs to strings.
df[PATIENT_COL] = df[PATIENT_COL].astype(str)
df[FOLDER_COL] = df[FOLDER_COL].astype(str)

# The dataset already contains the binary label IDs:
# RV = 0
# LV = 1

if "label_id" not in df.columns:
    raise ValueError(
        "Expected 'label_id' column in the dataset."
    )

#Validation of numerical IDs before converting to integers
label_ids = pd.to_numeric(df["label_id"], errors="raise")

if not label_ids.isin([0, 1]).all():
    raise ValueError("label_id must contain only 0 or 1, without missing values.")

#Check that the numerical IDs match clinical label names
label_names = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

expected_ids = label_names.map({"RV": 0, "LV": 1})

if expected_ids.isna().any():
    raise ValueError("Found missing or unexpected labels; epxected RV or LV.")

if not label_ids.eq(expected_ids).all():
    raise ValueError("Label mapping mismatch: expected RV=0 and LV=1.")


df["target"] = df["label_id"].astype(int)

# Verify that only RV/LV are present.
if not set(df["target"].unique()).issubset({0, 1}):
    raise ValueError(
        f"Unexpected label IDs: {df['target'].unique()}"
    )


# ============================================================
# CHECK SPLITS
# ============================================================

print("\nSplit distribution:")

print(
    df.groupby(SPLIT_COL)[PATIENT_COL]
      .nunique()
)

print("\nSeries distribution:")

print(
    df.groupby(SPLIT_COL)
      .size()
)


# Check that no patient occurs in multiple splits.

patient_split_counts = (
    df.groupby(PATIENT_COL)[SPLIT_COL]
      .nunique()
)

leaking_patients = patient_split_counts[
    patient_split_counts > 1
]

if len(leaking_patients) > 0:
    raise RuntimeError(
        f"Patient leakage detected: {len(leaking_patients)} patients "
        "appear in multiple splits."
    )

print("\nPatient leakage check: PASSED")


# ============================================================
# FIND HDF5 FILES
# ============================================================

print("\nFinding HDF5 files...")

h5_files = {
    Path(path).name.replace("_embed.hdf5", ""): Path(path)
    for path in glob.glob(str(EMBEDDING_DIR / "*_embed.hdf5"))
    if not path.endswith("_trim_embed.hdf5")
}

print("HDF5 files:", len(h5_files))


# ============================================================
# LOAD PRETRAINED ECHOFOCUS
# ============================================================

print("\nLoading pretrained EchoFocus CHD model...")

# These values come from:
# trained_models/EchoFocus_CHD/train_args.csv
#
# encoder_depth = 1
# clip_dropout = 0.5
# input_size = 768
# encoder_dim = 768
# output_size = 22

backbone = CustomTransformer(
    input_size=768,
    encoder_dim=768,
    n_encoder_layers=1,
    output_size=22,
    clip_dropout=0.5,
    tf_combine="avg",
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

backbone.load_state_dict(
    checkpoint["model_state_dict"]
)

backbone = backbone.to(DEVICE)

# Freeze EVERYTHING.
for param in backbone.parameters():
    param.requires_grad = False

# Disable dropout and other training-time behaviour.
backbone.eval()

print("Pretrained model loaded successfully.")


# ============================================================
# FUNCTION: LOAD SERIES EMBEDDING
# ============================================================

def load_series_embedding(folder, series_name):
    """
    Load one series from:

        HDF5
          └── patient folder
                └── series
                      └── emb

    Expected shape:
        (16, 768)
    """

    h5_path = h5_files.get(folder)

    if h5_path is None:
        raise FileNotFoundError(
            f"No HDF5 found for folder: {folder}"
        )

    with h5py.File(h5_path, "r") as f:

        if folder not in f:
            raise KeyError(
                f"Patient folder '{folder}' not found inside {h5_path}"
            )

        patient_group = f[folder]

        if series_name not in patient_group:
            raise KeyError(
                f"Series '{series_name}' not found in {h5_path}"
            )

        series_group = patient_group[series_name]

        if "emb" not in series_group:
            raise KeyError(
                f"No 'emb' dataset in series {series_name}"
            )

        emb = np.asarray(series_group["emb"][:])

    if emb.shape != (16, 768):
        raise ValueError(
            f"Unexpected embedding shape for {folder} / "
            f"{series_name}: {emb.shape}"
        )

    return torch.tensor(
        emb,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# DETERMINE SERIES COLUMN
# ============================================================

SERIES_COL = find_column([
    "series_name",
    "series",
    "series_id",
    "Series",
    "SeriesID",
    "dicom",
    "dicom_id",
])

if SERIES_COL is None:
    raise ValueError(
        "Could not find a series column in rvlv_dataset.csv. "
        f"Available columns: {df.columns.tolist()}"
    )

print("Series column:", SERIES_COL)


# ============================================================
# COMPUTE FROZEN ECHOFOCUS FEATURES
# ============================================================

if FEATURE_FILE.exists():

    print("\nFound cached frozen features:")
    print(FEATURE_FILE)

    feature_data = torch.load(
        FEATURE_FILE,
        map_location="cpu",
        weights_only=False,
    )

    X = feature_data["X"]
    y = feature_data["y"]
    metadata = feature_data["metadata"]

    print("Loaded cached features:", X.shape)

else:

    print("\nComputing frozen EchoFocus representations...")
    print("This runs the transformer ONCE per series.")

    features = []
    targets = []
    metadata = []

    with torch.no_grad():

        for i, row in df.iterrows():

            folder = row[FOLDER_COL]
            series_name = row[SERIES_COL]

            if i % 100 == 0:
                print(
                    f"Processing {i + 1}/{len(df)}"
                )

            emb = load_series_embedding(
                folder,
                series_name,
            )

            # EchoFocus embed() expects an iterable of tensors.
            #
            # Here the entire series consists of 16 clip embeddings.
            representation = backbone.embed(
                [emb]
            )

            if representation.shape != (768,):
                raise ValueError(
                    f"Unexpected representation shape: "
                    f"{representation.shape}"
                )

            features.append(
                representation.cpu()
            )

            targets.append(
                int(row["target"])
            )

            metadata.append({
                "patient_id": str(row[PATIENT_COL]),
                "folder": folder,
                "series": str(series_name),
                "label": int(row["target"]),
                "split": str(row[SPLIT_COL]),
            })

    X = torch.stack(features)
    y = torch.tensor(
        targets,
        dtype=torch.float32,
    )

    torch.save(
        {
            "X": X,
            "y": y,
            "metadata": metadata,
        },
        FEATURE_FILE,
    )

    print("\nSaved frozen features:")
    print(FEATURE_FILE)


if not torch.isfinite(y).all().item():
    raise ValueError("Fearture targets contain non-finite values.")

if not ((y == 0) | (y == 1)).all().item():
    raise ValueError("Feature targets must contain only 0 or 1.")

print("Feature target counts:", torch.unique(y, return_counts=True))


print("\nFeature matrix:", X.shape)
print("Labels:", y.shape)


# ============================================================
# RECONSTRUCT METADATA
# ============================================================

meta_df = pd.DataFrame(metadata)

meta_df["target"] = y.numpy().astype(int)

print("\nFeature labels:")
print(
    meta_df["target"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CREATE SPLIT MASKS
# ============================================================

train_mask = (
    meta_df["split"]
    .str.lower()
    .isin(["train", "training"])
)

val_mask = (
    meta_df["split"]
    .str.lower()
    .isin(["val", "valid", "validation"])
)

test_mask = (
    meta_df["split"]
    .str.lower()
    .isin(["test", "testing"])
)

print("\nFeature split sizes:")
print("Train:", train_mask.sum())
print("Val  :", val_mask.sum())
print("Test :", test_mask.sum())


if train_mask.sum() == 0:
    raise RuntimeError("No training samples found.")

if val_mask.sum() == 0:
    raise RuntimeError("No validation samples found.")

if test_mask.sum() == 0:
    raise RuntimeError("No test samples found.")


# ============================================================
# LINEAR PROBE
# ============================================================

classifier = nn.Linear(
    768,
    1,
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


train_X = X[train_mask.values]
train_y = y[train_mask.values]

val_X = X[val_mask.values]
val_y = y[val_mask.values]

test_X = X[test_mask.values]
test_y = y[test_mask.values]


train_dataset = TensorDataset(
    train_X,
    train_y,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold=0.5):

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = np.mean(
        predictions == y_true
    )

    # Confusion matrix components.
    tp = np.sum(
        (predictions == 1) &
        (y_true == 1)
    )

    tn = np.sum(
        (predictions == 0) &
        (y_true == 0)
    )

    fp = np.sum(
        (predictions == 1) &
        (y_true == 0)
    )

    fn = np.sum(
        (predictions == 0) &
        (y_true == 1)
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2

    # AUROC without requiring sklearn.
    try:
        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(
            y_true,
            probabilities,
        )
    except Exception:
        auroc = np.nan

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def predict(X_data):

    classifier.eval()

    with torch.no_grad():

        logits = classifier(
            X_data.to(DEVICE)
        ).squeeze(1)

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities.cpu().numpy()

#Added for evaluation and creating plots
def evaluate_split(X_data, y_data):
    "Evaluate one split using the current classifier."
    classifier.eval()

    with torch.no_grad():
        logits = classifier(
            X_data.to(DEVICE)
        ).squeeze(1)

        #BCEWithLogitsLoss before sigmoid below
        loss = criterion(
            logits,
            y_data.to(DEVICE),
        )

        probabilities = torch.sigmoid(logits).cpu().numpy()

    metrics = calculate_metrics(
        y_data.cpu().numpy(),
        probabilities,
    )

    return loss.item(), metrics

# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING LINEAR PROBE")
print("=" * 70)

best_val_auroc = -np.inf
best_state = None
epochs_without_improvement = 0

history = []

for epoch in range(1, EPOCHS + 1):

    classifier.train()

    running_loss = 0.0
    n_samples = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = classifier(
            batch_X
        ).squeeze(1)

        loss = criterion(
            logits,
            batch_y,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * len(batch_y)
        )

        n_samples += len(batch_y)

    #Average loss during parameters updating
    optimization_loss = running_loss / n_samples

    #Evaluation of both splots using completed epoch's classifier
    train_loss, train_metrics = evaluate_split(train_X, train_y)
    val_loss, val_metrics = evaluate_split(val_X, val_y)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_auroc": train_metrics["auroc"],
        "optimization_loss": optimization_loss,
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_auroc": val_metrics["auroc"],
        "val_sensitivity": val_metrics["sensitivity"],
        "val_specificity": val_metrics["specificity"],
    })

    pd.DataFrame(history).to_csv(
        RESULT_DIR / "training_history.csv",
        index=False,
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f} | "
        f"train accuracy {train_metrics['accuracy']:.4f} | "
        f"val accuracy {val_metrics['accuracy']:.4f} | "
        #f"loss {train_loss:.4f} | " 
        f"train AUROC {train_metrics['auroc']:.4f} | "
        f"val AUROC {val_metrics['auroc']:.4f}"
    )

    # Early stopping based on validation AUROC.
    if val_metrics["auroc"] > best_val_auroc:

        best_val_auroc = val_metrics["auroc"]

        best_state = {
            key: value.cpu().clone()
            for key, value
            in classifier.state_dict().items()
        }

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": best_state,
                "best_val_auroc": best_val_auroc,
                "epoch": epoch,
            },
            BEST_MODEL_FILE,
        )

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}."
            )

            break


# ============================================================
# LOAD BEST CLASSIFIER
# ============================================================

classifier.load_state_dict(
    best_state
)

classifier = classifier.to(DEVICE)


# ============================================================
# SERIES-LEVEL EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("SERIES-LEVEL TEST RESULTS")
print("=" * 70)

test_prob = predict(test_X)

series_metrics = calculate_metrics(
    test_y.numpy(),
    test_prob,
)

for key, value in series_metrics.items():
    print(f"{key:20s}: {value}")


# ============================================================
# PATIENT-LEVEL EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("PATIENT-LEVEL TEST RESULTS")
print("=" * 70)

test_meta = meta_df.loc[
    test_mask
].copy()

test_meta["probability_lv"] = test_prob

patient_predictions = (
    test_meta
    .groupby("patient_id")
    .agg(
        probability_lv=(
            "probability_lv",
            "mean"
        ),
        label=(
            "label",
            "first"
        ),
        n_series=(
            "series",
            "count"
        ),
    )
    .reset_index()
)

patient_metrics = calculate_metrics(
    patient_predictions["label"].values,
    patient_predictions["probability_lv"].values,
)

print(
    "Test patients:",
    len(patient_predictions)
)

for key, value in patient_metrics.items():
    print(f"{key:20s}: {value}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

all_prob = np.zeros(len(meta_df))

all_prob[train_mask.values] = predict(
    train_X
)

all_prob[val_mask.values] = predict(
    val_X
)

all_prob[test_mask.values] = test_prob

meta_df["probability_lv"] = all_prob

meta_df["prediction"] = (
    meta_df["probability_lv"] >= 0.5
).astype(int)

meta_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_rows = []

for level, metrics in [
    ("series_test", series_metrics),
    ("patient_test", patient_metrics),
]:

    row = {
        "level": level,
        **metrics,
    }

    metrics_rows.append(row)

pd.DataFrame(
    metrics_rows
).to_csv(
    METRICS_FILE,
    index=False,
)


pd.DataFrame(
    history
).to_csv(
    RESULT_DIR / "training_history.csv",
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print("\nBest validation AUROC:")
print(best_val_auroc)

print("\nSERIES TEST:")
print(series_metrics)

print("\nPATIENT TEST:")
print(patient_metrics)

print("\nFiles:")
print("Features   :", FEATURE_FILE)
print("Best model :", BEST_MODEL_FILE)
print("Metrics    :", METRICS_FILE)
print("Predictions:", PREDICTIONS_FILE)
print("=" * 70)
