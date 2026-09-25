from pathlib import Path

import h5py
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path.home() / "echofocus" / "ventricle_project"

LABEL_FILE = (
    BASE_DIR
    / "metadata"
    / "UCG lista enkammarhjärta.xlsx"
)

MAPPING_FILE = (
    BASE_DIR
    / "metadata"
    / "patient_id_folder_num.csv"
)

EMBEDDING_DIR = (
    BASE_DIR
    / "embeddings"
    / "dicom"
)

OUTPUT_DIR = (
    BASE_DIR
    / "splits"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "rvlv_dataset.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

EXPECTED_SHAPE = (16, 768)

LABEL_MAP = {
    "RV": 0,
    "LV": 1,
}


# ============================================================
# 1. LOAD CLINICAL LABELS
# ============================================================

print("=" * 70)
print("1. LOADING CLINICAL LABELS")
print("=" * 70)

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

print(f"Excel rows: {len(labels)}")

print("\nLabels:")
print(labels.value_counts())


# ============================================================
# 2. LOAD MAPPING
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING PATIENT MAPPING")
print("=" * 70)

mapping = pd.read_csv(
    MAPPING_FILE
)

print("Columns:")
print(mapping.columns.tolist())

required = {
    "folder",
    "index",
    "patient_id",
}

missing = required - set(mapping.columns)

if missing:
    raise ValueError(
        f"Missing mapping columns: {missing}"
    )


mapping = mapping[
    [
        "folder",
        "index",
        "patient_id",
    ]
].copy()


# Keep these as strings.
mapping["folder"] = (
    mapping["folder"]
    .astype(str)
    .str.strip()
)

mapping["patient_id"] = (
    mapping["patient_id"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 3. FIX / VALIDATE INDEX
# ============================================================

print("\n" + "=" * 70)
print("3. VALIDATING MAPPING")
print("=" * 70)

mapping["index"] = pd.to_numeric(
    mapping["index"],
    errors="coerce",
)

missing_index = mapping["index"].isna()

print(
    f"Missing indices: "
    f"{missing_index.sum()}"
)

# According to the mapping structure, the one blank index
# corresponds to the first Excel row, i.e. index 0.
if missing_index.sum() == 1:

    if 0 in set(
        mapping.loc[
            ~missing_index,
            "index"
        ].astype(int)
    ):
        raise ValueError(
            "Cannot safely assign the missing index: "
            "index 0 already exists."
        )

    mapping.loc[
        missing_index,
        "index"
    ] = 0


elif missing_index.sum() > 1:

    raise ValueError(
        "More than one missing mapping index."
    )


mapping["index"] = (
    mapping["index"]
    .astype(int)
)


# Check index range.
if mapping["index"].min() < 0:
    raise ValueError(
        "Negative mapping index found."
    )

if mapping["index"].max() >= len(labels):
    raise ValueError(
        "Mapping index exceeds number of Excel labels."
    )


# Check uniqueness.
if mapping["index"].duplicated().any():
    raise ValueError(
        "Duplicate mapping indices found."
    )

if mapping["patient_id"].duplicated().any():
    raise ValueError(
        "Duplicate patient_id values found."
    )

if mapping["folder"].duplicated().any():
    raise ValueError(
        "Duplicate folder values found."
    )


# ============================================================
# 4. MAP LABELS
# ============================================================

print("\n" + "=" * 70)
print("4. MAPPING LABELS TO PATIENTS")
print("=" * 70)

mapping["label"] = mapping["index"].apply(
    lambda i: labels.iloc[i]
)

print(
    mapping[
        [
            "index",
            "patient_id",
            "label",
            "folder",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nMapped patient labels:")
print(
    mapping["label"].value_counts()
)


# ============================================================
# 5. FIND HDF5 FILES
# ============================================================

print("\n" + "=" * 70)
print("5. SCANNING HDF5 FILES")
print("=" * 70)

hdf5_files = sorted(
    EMBEDDING_DIR.glob("*_embed.hdf5")
)

print(
    f"HDF5 files found: "
    f"{len(hdf5_files)}"
)

if not hdf5_files:
    raise FileNotFoundError(
        f"No HDF5 files found in:\n"
        f"{EMBEDDING_DIR}"
    )


# HDF5 lookup:
#
# folder -> path
#
hdf5_lookup = {}

for path in hdf5_files:

    folder = path.name.replace(
        "_embed.hdf5",
        ""
    )

    hdf5_lookup[folder] = path


# ============================================================
# 6. COMPARE HDF5 AND MAPPING
# ============================================================

mapping_folders = set(
    mapping["folder"]
)

hdf5_folders = set(
    hdf5_lookup.keys()
)

missing_hdf5 = (
    mapping_folders
    - hdf5_folders
)

extra_hdf5 = (
    hdf5_folders
    - mapping_folders
)


print(
    f"\nMapped folders without HDF5: "
    f"{len(missing_hdf5)}"
)

for folder in sorted(missing_hdf5):

    row = mapping[
        mapping["folder"] == folder
    ].iloc[0]

    print(
        f"  {folder}"
        f" | patient_id={row['patient_id']}"
        f" | index={row['index']}"
        f" | label={row['label']}"
    )


print(
    f"\nHDF5 folders without mapping: "
    f"{len(extra_hdf5)}"
)

for folder in sorted(extra_hdf5):

    print(
        f"  {folder}"
    )


# ============================================================
# 7. READ HDF5 SERIES
# ============================================================

print("\n" + "=" * 70)
print("6. READING HDF5 SERIES")
print("=" * 70)

records = []

patients_read = 0
patients_missing_hdf5 = 0
series_read = 0


for _, row in mapping.iterrows():

    folder = row["folder"]
    patient_id = row["patient_id"]
    index = row["index"]
    label = row["label"]


    # --------------------------------------------------------
    # HDF5 missing
    # --------------------------------------------------------

    if folder not in hdf5_lookup:

        patients_missing_hdf5 += 1
        continue


    h5_path = hdf5_lookup[folder]


    # --------------------------------------------------------
    # Open HDF5
    # --------------------------------------------------------

    with h5py.File(
        h5_path,
        "r"
    ) as f:

        # IMPORTANT:
        #
        # The top-level HDF5 group is `folder`.
        #
        # Example:
        #
        # folder:
        # bfr2loc_20260213_070102286_191_191
        #
        # HDF5:
        # bfr2loc_20260213_070102286_191_191_embed.hdf5
        #
        # Structure:
        #
        # bfr2loc_20260213_070102286_191_191/
        #     tp...dcm/
        #         emb
        #

        if folder not in f:

            raise ValueError(
                "\nHDF5 structure does not match mapping.\n"
                f"File: {h5_path}\n"
                f"Expected top-level group: {folder}\n"
                f"Found: {list(f.keys())}"
            )


        patient_group = f[folder]


        # ----------------------------------------------------
        # Each child is a series
        # ----------------------------------------------------

        for series_name in patient_group.keys():

            series_group = (
                patient_group[series_name]
            )


            if "emb" not in series_group:

                print(
                    "\nWARNING: no embedding:"
                )

                print(
                    f"Patient: {patient_id}"
                )

                print(
                    f"Series: {series_name}"
                )

                continue


            emb = series_group["emb"]


            # ------------------------------------------------
            # Check embedding shape
            # ------------------------------------------------

            if emb.shape != EXPECTED_SHAPE:

                raise ValueError(
                    "\nWrong embedding shape.\n"
                    f"Patient: {patient_id}\n"
                    f"Folder: {folder}\n"
                    f"Series: {series_name}\n"
                    f"Shape: {emb.shape}\n"
                    f"Expected: {EXPECTED_SHAPE}"
                )


            # ------------------------------------------------
            # Save metadata
            #
            # We do NOT copy the embedding into the CSV.
            # The Dataset/DataLoader will load it later.
            # ------------------------------------------------

            records.append(
                {
                    "PatientID": patient_id,
                    "folder": folder,
                    "index": index,
                    "series_name": series_name,
                    "hdf5_path": str(h5_path),
                    "label": label,
                }
            )

            series_read += 1


    patients_read += 1


print(
    f"\nPatients read: "
    f"{patients_read}"
)

print(
    f"Patients without HDF5: "
    f"{patients_missing_hdf5}"
)

print(
    f"Series read: "
    f"{series_read}"
)


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("7. CREATING DATASET")
print("=" * 70)

df = pd.DataFrame(
    records
)

if df.empty:
    raise RuntimeError(
        "No valid series were found."
    )


# ============================================================
# 9. VERIFY PATIENT CONSISTENCY
# ============================================================

patient_labels = (
    df[
        [
            "PatientID",
            "label",
        ]
    ]
    .drop_duplicates()
)

if (
    patient_labels
    .groupby("PatientID")
    .size()
    .gt(1)
    .any()
):
    raise ValueError(
        "A patient has multiple labels."
    )


# ============================================================
# 10. REMOVE BV
# ============================================================

print("\n" + "=" * 70)
print("8. RV VS LV")
print("=" * 70)

print("\nBefore removing BV:")

print(
    patient_labels["label"]
    .value_counts()
)


df = df[
    df["label"].isin(
        [
            "RV",
            "LV",
        ]
    )
].copy()


# Encode labels.
df["label_id"] = (
    df["label"]
    .map(LABEL_MAP)
    .astype(int)
)


patient_df = (
    df[
        [
            "PatientID",
            "label",
            "label_id",
        ]
    ]
    .drop_duplicates()
)


print("\nAfter removing BV:")

print(
    patient_df["label"]
    .value_counts()
)

print(
    f"\nRV/LV patients: "
    f"{len(patient_df)}"
)

print(
    f"RV/LV series: "
    f"{len(df)}"
)


# ============================================================
# 11. PATIENT-LEVEL SPLIT
# ============================================================

print("\n" + "=" * 70)
print("9. PATIENT-LEVEL TRAIN / VAL / TEST SPLIT")
print("=" * 70)

patients = (
    patient_df["PatientID"]
    .to_numpy()
)

patient_y = (
    patient_df["label_id"]
    .to_numpy()
)


# 70% train, 30% temporary
train_patients, temp_patients, \
train_y, temp_y = train_test_split(

    patients,

    patient_y,

    test_size=0.30,

    stratify=patient_y,

    random_state=RANDOM_STATE,
)


# 50% of temporary = 15% total validation
# 50% of temporary = 15% total test
val_patients, test_patients = (
    train_test_split(

        temp_patients,

        test_size=0.50,

        stratify=temp_y,

        random_state=RANDOM_STATE,
    )
)


train_set = set(
    train_patients
)

val_set = set(
    val_patients
)

test_set = set(
    test_patients
)


def get_split(patient_id):

    if patient_id in train_set:
        return "train"

    if patient_id in val_set:
        return "val"

    if patient_id in test_set:
        return "test"

    raise ValueError(
        f"Patient not assigned to split: "
        f"{patient_id}"
    )


df["split"] = (
    df["PatientID"]
    .apply(get_split)
)


# ============================================================
# 12. CHECK LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("10. CHECKING PATIENT LEAKAGE")
print("=" * 70)

assert train_set.isdisjoint(
    val_set
)

assert train_set.isdisjoint(
    test_set
)

assert val_set.isdisjoint(
    test_set
)


for split in (
    "train",
    "val",
    "test",
):

    patients_in_split = set(
        df.loc[
            df["split"] == split,
            "PatientID",
        ]
    )

    print(
        f"{split}: "
        f"{len(patients_in_split)} patients"
    )


print(
    "\n✓ No patient leakage."
)


# ============================================================
# 13. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("11. CLASS DISTRIBUTION")
print("=" * 70)


for split in (
    "train",
    "val",
    "test",
):

    subset = df[
        df["split"] == split
    ]

    patients_subset = (
        subset[
            [
                "PatientID",
                "label",
            ]
        ]
        .drop_duplicates()
    )


    print(
        f"\n{split.upper()}"
    )

    print(
        f"Patients: {len(patients_subset)}"
    )

    print(
        f"Series:   {len(subset)}"
    )

    print(
        patients_subset[
            "label"
        ]
        .value_counts()
        .sort_index()
    )


# ============================================================
# 14. SERIES PER PATIENT
# ============================================================

print("\n" + "=" * 70)
print("12. SERIES PER PATIENT")
print("=" * 70)

series_counts = (
    df.groupby(
        "PatientID"
    )
    .size()
)

print(
    series_counts.describe()
)

print(
    f"\nMaximum series per patient: "
    f"{series_counts.max()}"
)


# ============================================================
# 15. SAVE
# ============================================================

print("\n" + "=" * 70)
print("13. SAVING")
print("=" * 70)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"\nSaved:\n{OUTPUT_FILE}"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"\nPatients: "
    f"{df['PatientID'].nunique()}"
)

print(
    f"Series:   "
    f"{len(df)}"
)

print("\nLabels:")

print(
    df[
        [
            "PatientID",
            "label",
        ]
    ]
    .drop_duplicates()
    ["label"]
    .value_counts()
)

print("\nSplits:")

print(
    df[
        [
            "PatientID",
            "split",
        ]
    ]
    .drop_duplicates()
    ["split"]
    .value_counts()
)

print("\nExample:")

print(
    df[
        [
            "PatientID",
            "folder",
            "series_name",
            "label",
            "label_id",
            "split",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nDONE.")
