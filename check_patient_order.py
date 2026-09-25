from pathlib import Path
import h5py
import pandas as pd


BASE_DIR = Path.home() / "echofocus" / "ventricle_project"

LABEL_FILE = BASE_DIR / "metadata" / "UCG lista enkammarhjärta.xlsx"
EMBEDDING_DIR = BASE_DIR / "embeddings" / "dicom"


# ------------------------------------------------------------
# Excel
# ------------------------------------------------------------

labels_df = pd.read_excel(
    LABEL_FILE,
    header=None,
)

labels = (
    labels_df.iloc[:, 0]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("=" * 70)
print("EXCEL")
print("=" * 70)

print("Number of rows:", len(labels))

print("\nFirst 20 labels:")
for i, label in enumerate(labels.iloc[:20]):
    print(f"{i:4d} -> {label}")


# ------------------------------------------------------------
# HDF5
# ------------------------------------------------------------

hdf5_files = sorted(
    EMBEDDING_DIR.glob("*_embed.hdf5")
)

print("\n" + "=" * 70)
print("HDF5")
print("=" * 70)

print(
    "Number of HDF5 files:",
    len(hdf5_files)
)

print("\nFirst 20 HDF5 files:")

for i, path in enumerate(hdf5_files[:20]):

    patient_id = path.name.replace(
        "_embed.hdf5",
        "",
    )

    print(
        f"{i:4d} -> {patient_id}"
    )
