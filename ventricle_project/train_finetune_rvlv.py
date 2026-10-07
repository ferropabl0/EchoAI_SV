import os
import glob
import random
import sys
from pathlib import Path

# Make EchoFocus repository root importable.
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import CustomTransformer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path.home() / "echofocus" / "ventricle_project"

DATASET_CSV = BASE_DIR / "splits" / "rvlv_dataset.csv"
EMBEDDING_DIR = BASE_DIR / "embeddings" / "dicom"

CHECKPOINT = (
    Path.home()
    / "echofocus"
    / "trained_models"
    / "EchoFocus_CHD"
    / "best_checkpoint.pt"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "finetune_rvlv"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_FILE = RESULT_DIR / "best_finetuned_model.pt"
METRICS_FILE = RESULT_DIR / "metrics.csv"
PREDICTIONS_FILE = RESULT_DIR / "predictions.csv"
HISTORY_FILE = RESULT_DIR / "training_history.csv"


# Training
SEED = 42
EPOCHS = 30
BATCH_SIZE = 1

BACKBONE_LR = 1e-6
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-2

PATIENCE = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("EchoFocus — RV vs LV Fine-Tuning")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset:")
print(DATASET_CSV)

df = pd.read_csv(
    DATASET_CSV
)

print("\nRows:", len(df))

print(
    "Columns:",
    df.columns.tolist()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "PatientID",
    "folder",
    "series_name",
    "label",
    "label_id",
    "split",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


df["PatientID"] = (
    df["PatientID"]
    .astype(str)
)

df["folder"] = (
    df["folder"]
    .astype(str)
)

df["series_name"] = (
    df["series_name"]
    .astype(str)
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


# RV = 0
# LV = 1
df["target"] = (
    df["label_id"]
    .astype(int)
)


if not set(
    df["target"].unique()
).issubset({0, 1}):

    raise ValueError(
        f"Unexpected labels: "
        f"{df['target'].unique()}"
    )

# ============================================================
# CHECK PATIENT LEAKAGE
# ============================================================

patient_split_counts = (
    df.groupby("PatientID")["split"]
    .nunique()
)

leaking = patient_split_counts[
    patient_split_counts > 1
]

if len(leaking) > 0:

    raise RuntimeError(
        f"Patient leakage detected: "
        f"{len(leaking)} patients."
    )

print(
    "\nPatient leakage check: PASSED"
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

print("\nDataset splits:")

for split in ["train", "val", "test"]:

    subset = df[
        df["split"].str.lower() == split
    ]

    print(
        f"{split:5s}: "
        f"{len(subset):5d} series, "
        f"{subset['PatientID'].nunique():3d} patients, "
        f"LV={(subset['target'] == 1).sum():4d}, "
        f"RV={(subset['target'] == 0).sum():4d}"
    )


# ============================================================
# FIND HDF5 FILES
# ============================================================

print("\nFinding HDF5 files...")

h5_files = {}

for path in glob.glob(
    str(EMBEDDING_DIR / "*_embed.hdf5")
):

    if path.endswith(
        "_trim_embed.hdf5"
    ):
        continue

    filename = Path(path).name

    folder = filename.replace(
        "_embed.hdf5",
        ""
    )

    h5_files[folder] = Path(path)

print(
    "HDF5 files:",
    len(h5_files)
)


# ============================================================
# DATASET
# ============================================================

class EchoSeriesDataset(Dataset):

    def __init__(
        self,
        dataframe,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        folder = row["folder"]
        series_name = row["series_name"]

        h5_path = h5_files.get(folder)

        if h5_path is None:

            raise FileNotFoundError(
                f"No HDF5 for folder: {folder}"
            )

        with h5py.File(
            h5_path,
            "r"
        ) as f:

            if folder not in f:

                raise KeyError(
                    f"Folder {folder} "
                    f"not found in {h5_path}"
                )

            patient_group = f[folder]

            if series_name not in patient_group:

                raise KeyError(
                    f"Series {series_name} "
                    f"not found in {h5_path}"
                )

            series_group = (
                patient_group[series_name]
            )

            if "emb" not in series_group:

                raise KeyError(
                    f"No emb dataset in "
                    f"{series_name}"
                )

            emb = np.asarray(
                series_group["emb"][:],
                dtype=np.float32
            )

        if emb.shape != (16, 768):

            raise ValueError(
                f"Unexpected embedding shape "
                f"{emb.shape} for "
                f"{folder}/{series_name}"
            )

        x = torch.tensor(
            emb,
            dtype=torch.float32
        )

        y = torch.tensor(
            float(row["target"]),
            dtype=torch.float32
        )

        return (
            x,
            y,
            str(row["PatientID"]),
            folder,
            series_name,
        )


# ============================================================
# CREATE DATASETS
# ============================================================

train_df = df[
    df["split"].str.lower() == "train"
].copy()

val_df = df[
    df["split"].str.lower() == "val"
].copy()

test_df = df[
    df["split"].str.lower() == "test"
].copy()


train_dataset = EchoSeriesDataset(
    train_df
)

val_dataset = EchoSeriesDataset(
    val_df
)

test_dataset = EchoSeriesDataset(
    test_df
)


train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

train_eval_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD PRETRAINED ECHOFOCUS
# ============================================================

print("\nLoading pretrained EchoFocus CHD...")


model = CustomTransformer(
    input_size=768,
    encoder_dim=768,
    n_encoder_layers=1,
    output_size=22,
    clip_dropout=0.0,
    tf_combine="avg",
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "Pretrained checkpoint loaded."
)


# ============================================================
# REPLACE CLASSIFICATION HEAD
# ============================================================

model.ff = nn.Linear(
    768,
    1
)

model = model.to(DEVICE)


print(
    "\nNew classification head:",
    model.ff
)


# ============================================================
# PARAMETER GROUPS
# ============================================================

backbone_parameters = []

for name, param in model.named_parameters():

    if name.startswith("ff."):

        continue

    backbone_parameters.append(
        param
    )


classifier_parameters = list(
    model.ff.parameters()
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR,
        },
        {
            "params": classifier_parameters,
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY
)


criterion = nn.BCEWithLogitsLoss()


print("\nLearning rates:")
print(
    "Backbone  :",
    BACKBONE_LR
)
print(
    "Classifier:",
    CLASSIFIER_LR
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        roc_auc_score,
        confusion_matrix,
    )

    y_true = np.asarray(
        y_true
    ).astype(int)

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            predictions
        )
    )

    try:

        auroc = roc_auc_score(
            y_true,
            probabilities
        )

    except ValueError:

        auroc = np.nan

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else np.nan
    )

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


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    all_probabilities = []
    all_labels = []
    all_patients = []
    all_folders = []
    all_series = []

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():

        for (
            x,
            y,
            patient,
            folder,
            series,
        ) in loader:

            # DataLoader adds batch dimension:
            # x = (1, 16, 768)
            x = x[0].to(DEVICE)

            y = y.to(DEVICE)

            logits = model(
                [x]
            ).reshape(-1)

            #BCEWithLogitsLoss before sigmoid
            loss = criterion(
                logits,
                y.reshape(-1)
            )

            probability = (
                torch.sigmoid(
                    logits
                )
                .item()
            )

            label = (
                int(y.item())
            )

            all_probabilities.append(
                probability
            )

            all_labels.append(
                label
            )

            all_patients.append(
                patient[0]
            )

            all_folders.append(
                folder[0]
            )

            all_series.append(
                series[0]
            )

            total_loss += (
                loss.item()
            )

            total_n += 1

    metrics = calculate_metrics(
        all_labels,
        all_probabilities
    )

    predictions = pd.DataFrame({
        "PatientID": all_patients,
        "folder": all_folders,
        "series_name": all_series,
        "label": all_labels,
        "probability_lv": all_probabilities,
    })

    predictions["prediction"] = (
        predictions["probability_lv"]
        >= 0.5
    ).astype(int)

    return (
        total_loss / total_n,
        metrics,
        predictions,
    )


# ============================================================
# PATIENT-LEVEL METRICS
# ============================================================

def patient_level_metrics(
    predictions
):

    patient_predictions = (
        predictions
        .groupby("PatientID")
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
                "series_name",
                "count"
            ),
        )
        .reset_index()
    )

    metrics = calculate_metrics(
        patient_predictions["label"],
        patient_predictions[
            "probability_lv"
        ],
    )

    return (
        metrics,
        patient_predictions,
    )


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

best_val_auroc = -np.inf
best_state = None
epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    n_samples = 0

    for (
        x,
        y,
        patient,
        folder,
        series,
    ) in train_loader:

        # Remove batch dimension.
        x = x[0].to(DEVICE)

        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            [x]
        ).reshape(-1)

        loss = criterion(
            logits,
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

        n_samples += 1

    optimization_loss = running_loss / n_samples

    #Evaluation of both splits using epoch's model
    train_loss, train_metrics, _ = evaluate(train_eval_loader)
    val_loss, val_metrics, _ = evaluate(val_loader)


    print(
        f"Epoch {epoch:02d} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f} | "
        f"train accuracy {train_metrics['accuracy']:.4f} | "
        f"val accuracy {val_metrics['accuracy']:.4f} | "
        f"train AUROC {train_metrics['auroc']:.4f} | "
        f"val AUROC "
        f"{val_metrics['auroc']:.4f}"
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "optimization_loss": optimization_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_auroc": train_metrics["auroc"],
        "val_loss": val_loss,
        "val_accuracy": (
            val_metrics["accuracy"]
        ),
        "val_balanced_accuracy": (
            val_metrics[
                "balanced_accuracy"
            ]
        ),
        "val_auroc": (
            val_metrics["auroc"]
        ),
        "val_sensitivity": (
            val_metrics["sensitivity"]
        ),
        "val_specificity": (
            val_metrics["specificity"]
        ),
    })

    pd.DataFrame(history).to_csv(
        HISTORY_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if (
        val_metrics["auroc"]
        > best_val_auroc
    ):

        best_val_auroc = (
            val_metrics["auroc"]
        )

        best_state = {
            key: value.detach()
            .cpu()
            .clone()
            for key, value
            in model.state_dict()
            .items()
        }

        torch.save(
            {
                "model_state_dict":
                    best_state,

                "best_val_auroc":
                    best_val_auroc,

                "epoch":
                    epoch,

                "backbone_lr":
                    BACKBONE_LR,

                "classifier_lr":
                    CLASSIFIER_LR,
            },
            BEST_MODEL_FILE
        )

        epochs_without_improvement = 0

        print(
            "  -> New best model"
        )

    else:

        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is None:

    raise RuntimeError(
        "No best model was saved."
    )

model.load_state_dict(
    best_state
)

model = model.to(DEVICE)


# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)


test_loss, series_metrics, test_predictions = (
    evaluate(test_loader)
)


print("\nSeries level:")
print(
    "Test loss:",
    test_loss
)

for key, value in series_metrics.items():

    print(
        f"{key:20s}: {value}"
    )


patient_metrics, patient_predictions = (
    patient_level_metrics(
        test_predictions
    )
)


print("\nPatient level:")
print(
    "Test patients:",
    len(patient_predictions)
)

for key, value in patient_metrics.items():

    print(
        f"{key:20s}: {value}"
    )


# ============================================================
# SAVE
# ============================================================

test_predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)

patient_predictions.to_csv(
    RESULT_DIR
    / "patient_predictions.csv",
    index=False
)

pd.DataFrame(
    history
).to_csv(
    HISTORY_FILE,
    index=False
)


metrics_df = pd.DataFrame([
    {
        "level": "series_test",
        **series_metrics,
    },
    {
        "level": "patient_test",
        **patient_metrics,
    },
])

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Best validation AUROC:",
    best_val_auroc
)

print(
    "\nBest model:",
    BEST_MODEL_FILE
)

print(
    "Metrics:",
    METRICS_FILE
)

print(
    "Predictions:",
    PREDICTIONS_FILE
)

print(
    "History:",
    HISTORY_FILE
)

print("=" * 70)
