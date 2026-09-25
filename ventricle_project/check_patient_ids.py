import os
import re
import csv
from collections import Counter

ROOT = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
OUTPUT = "/home/echo2026/echofocus/ventricle_project/patient_id_audit.csv"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_index(folder_name):
    """
    Extract the final numeric index from normal disk_* folder names.

    Example:
        disk_20260211_133821609_1_1_1 -> 1
        disk_20260211_223654715_19_19_101 -> 101

    bfr2loc folders intentionally return None.
    """
    m = re.match(r"^disk_\d{8}_\d{9}_\d+_\d+_(\d+)$", folder_name)
    if m:
        return int(m.group(1))
    return None


def find_first_dicom(folder):
    """
    Find the first real .dcm file recursively.
    Hidden AppleDouble files (._*) are ignored.
    """
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for f in sorted(files):
            if f.startswith("."):
                continue
            if not f.lower().endswith(".dcm"):
                continue

            return os.path.join(root, f)

    return None


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

print("=" * 78)
print("PATIENT ID AUDIT")
print("=" * 78)

print()
print("Root:")
print(ROOT)
print()

folders = [
    d for d in os.listdir(ROOT)
    if os.path.isdir(os.path.join(ROOT, d))
    and not d.startswith(".")
]

folders.sort()

print(f"Total patient folders: {len(folders)}")
print()

rows = []
errors = []

try:
    import pydicom
except ImportError:
    print("ERROR: pydicom is not installed in this environment.")
    raise


for folder_name in folders:

    folder_path = os.path.join(ROOT, folder_name)

    index = extract_index(folder_name)

    dicom_path = find_first_dicom(folder_path)

    patient_id = None
    patient_name = None
    study_uid = None

    if dicom_path is None:
        errors.append((folder_name, "No DICOM file found"))
    else:
        try:
            ds = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=True,
                force=True
            )

            patient_id = str(getattr(ds, "PatientID", "")).strip()
            patient_name = str(getattr(ds, "PatientName", "")).strip()
            study_uid = str(
                getattr(ds, "StudyInstanceUID", "")
            ).strip()

        except Exception as e:
            errors.append((folder_name, str(e)))

    rows.append({
        "folder": folder_name,
        "index": index,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "study_instance_uid": study_uid,
        "dicom_path": dicom_path or "",
    })


# ------------------------------------------------------------
# PatientID analysis
# ------------------------------------------------------------

patient_ids = [
    int(r["patient_id"])
    for r in rows
    if r["patient_id"] and r["patient_id"].isdigit()
]

counter = Counter(patient_ids)

duplicate_patient_ids = sorted(
    pid for pid, count in counter.items()
    if count > 1
)

present_ids = set(patient_ids)

expected_ids = set(range(1, 401))

missing_patient_ids = sorted(expected_ids - present_ids)
extra_patient_ids = sorted(present_ids - expected_ids)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(f"Total folders:                 {len(folders)}")
print(f"Folders with numeric PatientID: {len(patient_ids)}")
print(f"Unique PatientIDs:             {len(present_ids)}")
print(f"Read errors / missing DICOM:   {len(errors)}")

print()
print(f"Missing PatientIDs from 1-400: {missing_patient_ids}")
print(f"Extra PatientIDs outside 1-400: {extra_patient_ids}")
print(f"Duplicate PatientIDs:          {duplicate_patient_ids}")

print()

# ------------------------------------------------------------
# Index analysis
# ------------------------------------------------------------

indices = [
    r["index"]
    for r in rows
    if r["index"] is not None
]

index_set = set(indices)

expected_indices = set(range(1, 399))

missing_indices = sorted(expected_indices - index_set)

duplicate_indices = sorted(
    idx for idx, count in Counter(indices).items()
    if count > 1
)

print("=" * 78)
print("FOLDER INDEX")
print("=" * 78)

print(f"Indexed folders:               {len(indices)}")
print(f"Unique indices:                {len(index_set)}")
print(f"Missing indices 1-398:         {missing_indices}")
print(f"Duplicate indices:             {duplicate_indices}")

print()

# ------------------------------------------------------------
# Important PatientID / folder mapping
# ------------------------------------------------------------

print("=" * 78)
print("IMPORTANT PATIENT IDs")
print("=" * 78)

for pid in [19, 191, 192, 272]:
    matches = [
        r for r in rows
        if r["patient_id"] == str(pid)
    ]

    print()
    print(f"PatientID {pid}:")

    if not matches:
        print("  NOT FOUND")
    else:
        for r in matches:
            print(
                f"  index={str(r['index']):>3} | "
                f"{r['folder']}"
            )

print()

# ------------------------------------------------------------
# Print suspicious / anomalous folders
# ------------------------------------------------------------

print("=" * 78)
print("ANOMALOUS FOLDERS")
print("=" * 78)

for r in rows:
    if r["index"] is None:
        print(
            f"NO INDEX | PatientID={r['patient_id']} | "
            f"{r['folder']}"
        )

print()

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

fieldnames = [
    "folder",
    "index",
    "patient_id",
    "patient_name",
    "study_instance_uid",
    "dicom_path",
]

with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("=" * 78)
print("OUTPUT")
print("=" * 78)

print(f"CSV saved to:")
print(OUTPUT)

print()

if errors:
    print("DICOM errors:")
    for folder, error in errors:
        print(f"  {folder}: {error}")

print()
print("=" * 78)
