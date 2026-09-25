import os
import csv
import glob
import h5py
import numpy as np
import torch

from echofocus import EchoFocus, load_model_and_random_state


# ============================================================
# CONFIGURATION
# ============================================================

EMBED_DIR = "ventricle_project/embeddings/dicom"

RESULT_DIR = "ventricle_project/results"

RESULT_CSV = os.path.join(
    RESULT_DIR,
    "echofocus_chd_predictions.csv"
)

ERROR_CSV = os.path.join(
    RESULT_DIR,
    "echofocus_chd_errors.csv"
)

CHECKPOINT = "./trained_models/EchoFocus_CHD/best_checkpoint.pt"


LABELS = [
    "ASD",
    "AnomCA",
    "BAV",
    "DORV",
    "DTGA",
    "Ebstein",
    "HLHS",
    "LSVC",
    "PAPVR",
    "PDA",
    "RAA",
    "Tri atresia",
    "Truncus arteriosus",
    "Single_V_Composite",
    "TOF_ADJ",
    "AVCD_ADJ",
    "VSD_ADJ",
    "Coarct_ADJ",
    "PULMONARY_ATRESIA_ADJ",
    "TAPVR_ADJ",
    "Composite_NonCritical",
    "Composite_Critical",
]


# ============================================================
# SETUP
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 70)
print("EchoFocus CHD — MASS INFERENCE")
print("=" * 70)

# ------------------------------------------------------------
# Find HDF5 files
# ------------------------------------------------------------

all_files = sorted(
    glob.glob(
        os.path.join(
            EMBED_DIR,
            "*_embed.hdf5"
        )
    )
)

# Never treat the old trim format as input.
h5_files = [
    f for f in all_files
    if not f.endswith("_trim_embed.hdf5")
]

print("\nHDF5 files found:", len(h5_files))


# ============================================================
# RESUME SUPPORT
# ============================================================

completed = set()

if os.path.exists(RESULT_CSV):

    with open(
        RESULT_CSV,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row.get("patient_id"):
                completed.add(row["patient_id"])

print("Already completed:", len(completed))
print("Remaining:", len(h5_files) - len(completed))


# ============================================================
# MODEL
# ============================================================

print("\nLoading EchoFocus_CHD...")

wrapper = EchoFocus(
    model_name="EchoFocus_CHD",
    dataset="outside",
    task="chd",
    test_only=True,
)

model, _, _, _, _ = wrapper._setup_model()

model, _, _, _, _ = load_model_and_random_state(
    CHECKPOINT,
    model
)

model.eval()

device = next(model.parameters()).device

print("Model device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)


# ============================================================
# CSV INITIALIZATION
# ============================================================

csv_columns = (
    ["patient_id", "n_series", "n_clips"]
    + [
        f"logit_{label}"
        for label in LABELS
    ]
    + [
        f"prob_{label}"
        for label in LABELS
    ]
)

if not os.path.exists(RESULT_CSV):

    with open(
        RESULT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_columns
        )

        writer.writeheader()


if not os.path.exists(ERROR_CSV):

    with open(
        ERROR_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "patient_id",
                "file",
                "error",
            ]
        )


# ============================================================
# PROCESS PATIENT
# ============================================================

processed = 0
failed = 0
skipped = 0

total = len(h5_files)

for i, h5_path in enumerate(h5_files, start=1):

    filename = os.path.basename(h5_path)

    patient_id = filename.replace(
        "_embed.hdf5",
        ""
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if patient_id in completed:

        skipped += 1

        print(
            f"[{i}/{total}] SKIP {patient_id}"
        )

        continue

    print()
    print("-" * 70)
    print(
        f"[{i}/{total}] {patient_id}"
    )

    try:

        # ====================================================
        # READ HDF5
        # ====================================================

        with h5py.File(h5_path, "r") as f:

            if patient_id not in f:

                raise RuntimeError(
                    f"Patient group '{patient_id}' not found"
                )

            patient_group = f[patient_id]

            series_names = sorted(
                patient_group.keys()
            )

            if len(series_names) == 0:

                raise RuntimeError(
                    "No series found"
                )

            study_embeddings = []

            for series_name in series_names:

                series_group = patient_group[
                    series_name
                ]

                if "emb" not in series_group:

                    raise RuntimeError(
                        f"{series_name}: missing emb dataset"
                    )

                emb = series_group["emb"][()]

                if emb.shape != (16, 768):

                    raise RuntimeError(
                        f"{series_name}: "
                        f"unexpected shape {emb.shape}"
                    )

                if not np.isfinite(emb).all():

                    raise RuntimeError(
                        f"{series_name}: "
                        "contains NaN or Inf"
                    )

                study_embeddings.append(
                    torch.from_numpy(emb)
                    .to(
                        device=device,
                        dtype=torch.float32
                    )
                )

        n_series = len(study_embeddings)

        n_clips = sum(
            x.shape[0]
            for x in study_embeddings
        )

        print(
            f"  Series: {n_series}"
        )

        print(
            f"  Clips:  {n_clips}"
        )

        # ====================================================
        # INFERENCE
        # ====================================================

        with torch.no_grad():

            logits = model(
                study_embeddings
            )

        logits = logits.detach().cpu()

        if logits.shape != (22,):

            raise RuntimeError(
                f"Unexpected model output shape "
                f"{tuple(logits.shape)}"
            )

        probabilities = torch.sigmoid(
            logits
        )

        logits_np = logits.numpy()
        probabilities_np = probabilities.numpy()

        # ====================================================
        # WRITE RESULT IMMEDIATELY
        # ====================================================

        row = {
            "patient_id": patient_id,
            "n_series": n_series,
            "n_clips": n_clips,
        }

        for label, value in zip(
            LABELS,
            logits_np
        ):

            row[
                f"logit_{label}"
            ] = float(value)

        for label, value in zip(
            LABELS,
            probabilities_np
        ):

            row[
                f"prob_{label}"
            ] = float(value)

        with open(
            RESULT_CSV,
            "a",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=csv_columns
            )

            writer.writerow(row)

        processed += 1

        print("  ✓ Prediction saved")

    except Exception as e:

        failed += 1

        print(
            f"  ✗ ERROR: {type(e).__name__}: {e}"
        )

        with open(
            ERROR_CSV,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    patient_id,
                    h5_path,
                    f"{type(e).__name__}: {e}",
                ]
            )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("MASS INFERENCE FINISHED")
print("=" * 70)

print("Total HDF5 files:", total)
print("Processed:", processed)
print("Skipped:", skipped)
print("Failed:", failed)

print()
print("Results:")
print(RESULT_CSV)

print()
print("Errors:")
print(ERROR_CSV)

print("=" * 70)
