#!/usr/bin/env python3

import os
import re
import csv
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"

EXPECTED_MIN = 1
EXPECTED_MAX = 398

OUTPUT_CSV = os.path.expanduser(
    "~/echofocus/ventricle_project/patient_folder_audit.csv"
)

# Expected normal folder pattern:
#
# disk_20260212_192059964_99_99_398
#
# We only care about the final integer.
NORMAL_PATTERN = re.compile(
    r"^disk_.*_(\d+)$"
)

# ============================================================
# HELPERS
# ============================================================

def is_hidden(name):
    return name.startswith(".")


def list_real_entries(path):
    """
    List entries while ignoring hidden files such as:
    .DS_Store
    ._something
    """
    try:
        return [
            name for name in os.listdir(path)
            if not is_hidden(name)
        ]
    except Exception as e:
        return None


def inspect_folder(folder_path):
    """
    Inspect whether the patient folder is empty and whether
    its DICOM directory exists/is empty.
    """

    result = {
        "folder_empty": False,
        "dicom_exists": False,
        "dicom_empty": False,
        "dicom_file_count": None,
        "error": "",
    }

    try:
        entries = list_real_entries(folder_path)

        if entries is None:
            result["error"] = "Could not list patient folder"
            return result

        result["folder_empty"] = len(entries) == 0

        # Look for DICOM directory, case-insensitively.
        dicom_dirs = [
            name for name in entries
            if name.lower() == "dicom"
            and os.path.isdir(os.path.join(folder_path, name))
        ]

        if not dicom_dirs:
            return result

        dicom_path = os.path.join(folder_path, dicom_dirs[0])
        result["dicom_exists"] = True

        dicom_entries = list_real_entries(dicom_path)

        if dicom_entries is None:
            result["error"] = "Could not list DICOM directory"
            return result

        result["dicom_empty"] = len(dicom_entries) == 0

        # Count actual files, not directories.
        file_count = 0

        for root, dirs, files in os.walk(dicom_path):
            dirs[:] = [d for d in dirs if not is_hidden(d)]

            for filename in files:
                if not is_hidden(filename):
                    file_count += 1

        result["dicom_file_count"] = file_count

    except Exception as e:
        result["error"] = repr(e)

    return result


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("PATIENT FOLDER AUDIT")
    print("=" * 70)
    print()
    print(f"Root directory:")
    print(ROOT)
    print()
    print(f"Expected indices: {EXPECTED_MIN}–{EXPECTED_MAX}")
    print()

    if not os.path.isdir(ROOT):
        print("ERROR: Root directory does not exist or is not accessible.")
        return

    # --------------------------------------------------------
    # Find top-level directories
    # --------------------------------------------------------

    try:
        top_entries = os.listdir(ROOT)
    except Exception as e:
        print(f"ERROR: Could not read root directory: {e}")
        return

    patient_dirs = []

    for name in top_entries:

        # Ignore hidden/system entries.
        if is_hidden(name):
            continue

        path = os.path.join(ROOT, name)

        if os.path.isdir(path):
            patient_dirs.append(name)

    patient_dirs.sort()

    print(f"Total top-level directories: {len(patient_dirs)}")
    print()

    # --------------------------------------------------------
    # Classify folders
    # --------------------------------------------------------

    index_to_folders = defaultdict(list)
    anomalous = []

    records = []

    for name in patient_dirs:

        path = os.path.join(ROOT, name)

        match = NORMAL_PATTERN.match(name)

        if match:
            index = int(match.group(1))
            name_type = "NORMAL"
            index_to_folders[index].append(name)
        else:
            index = ""
            name_type = "ANOMALOUS"
            anomalous.append(name)

        inspection = inspect_folder(path)

        records.append({
            "folder_name": name,
            "index": index,
            "name_type": name_type,
            "folder_empty": inspection["folder_empty"],
            "dicom_exists": inspection["dicom_exists"],
            "dicom_empty": inspection["dicom_empty"],
            "dicom_file_count": inspection["dicom_file_count"],
            "error": inspection["error"],
        })

    # --------------------------------------------------------
    # Missing indices
    # --------------------------------------------------------

    expected_indices = set(range(EXPECTED_MIN, EXPECTED_MAX + 1))
    found_indices = set(index_to_folders.keys())

    missing_indices = sorted(expected_indices - found_indices)

    # --------------------------------------------------------
    # Duplicate indices
    # --------------------------------------------------------

    duplicate_indices = {
        index: folders
        for index, folders in index_to_folders.items()
        if len(folders) > 1
    }

    # --------------------------------------------------------
    # Empty folders
    # --------------------------------------------------------

    empty_folders = [
        r["folder_name"]
        for r in records
        if r["folder_empty"]
    ]

    empty_dicom = [
        r["folder_name"]
        for r in records
        if r["dicom_exists"] and r["dicom_empty"]
    ]

    missing_dicom = [
        r["folder_name"]
        for r in records
        if not r["dicom_exists"]
    ]

    errors = [
        r for r in records
        if r["error"]
    ]

    # ========================================================
    # REPORT
    # ========================================================

    print("=" * 70)
    print("1. INDEX SUMMARY")
    print("=" * 70)

    print(f"Expected indices:       {EXPECTED_MAX}")
    print(f"Indices found:          {len(found_indices)}")
    print(f"Missing indices:        {len(missing_indices)}")
    print(f"Duplicate indices:      {len(duplicate_indices)}")
    print(f"Anomalous folders:      {len(anomalous)}")

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. MISSING INDICES")
    print("=" * 70)

    if missing_indices:
        print("Missing indices:")
        print(", ".join(map(str, missing_indices)))
    else:
        print("None. All indices 1–398 are present.")

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. DUPLICATE INDICES")
    print("=" * 70)

    if duplicate_indices:
        for index in sorted(duplicate_indices):
            print(f"\nIndex {index}:")
            for folder in duplicate_indices[index]:
                print(f"  - {folder}")
    else:
        print("No duplicate indices found.")

    # --------------------------------------------------------
    # Anomalous names
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("4. ANOMALOUS FOLDER NAMES")
    print("=" * 70)

    if anomalous:
        for folder in anomalous:
            print(f"  - {folder}")
    else:
        print("No anomalous folder names found.")

    # --------------------------------------------------------
    # Empty folders
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5. EMPTY PATIENT FOLDERS")
    print("=" * 70)

    if empty_folders:
        for folder in empty_folders:
            print(f"  - {folder}")
    else:
        print("No empty patient folders found.")

    # --------------------------------------------------------
    # Empty DICOM folders
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("6. EMPTY DICOM FOLDERS")
    print("=" * 70)

    if empty_dicom:
        for folder in empty_dicom:
            print(f"  - {folder}")
    else:
        print("No empty DICOM folders found.")

    # --------------------------------------------------------
    # Missing DICOM directory
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("7. PATIENT FOLDERS WITHOUT DICOM DIRECTORY")
    print("=" * 70)

    if missing_dicom:
        for folder in missing_dicom:
            print(f"  - {folder}")
    else:
        print("Every patient folder contains a DICOM directory.")

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("8. READ/ACCESS ERRORS")
    print("=" * 70)

    if errors:
        for r in errors:
            print(f"{r['folder_name']}: {r['error']}")
    else:
        print("No read/access errors.")

    # ========================================================
    # SPECIAL CHECK FOR KNOWN ANOMALOUS FOLDER
    # ========================================================

    known_anomalous = "bfr2loc_20260213_070102286_191_191"

    print()
    print("=" * 70)
    print("9. KNOWN ANOMALOUS FOLDER")
    print("=" * 70)

    if known_anomalous in patient_dirs:
        print(f"FOUND: {known_anomalous}")
        print("This folder does not match the normal disk_* naming pattern.")
    else:
        print(f"NOT FOUND: {known_anomalous}")

    # ========================================================
    # CSV
    # ========================================================

    print()
    print("=" * 70)
    print("10. WRITING CSV REPORT")
    print("=" * 70)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

        fieldnames = [
            "folder_name",
            "index",
            "name_type",
            "folder_empty",
            "dicom_exists",
            "dicom_empty",
            "dicom_file_count",
            "error",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow(r)

    print(f"CSV written to:")
    print(OUTPUT_CSV)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Top-level patient directories: {len(patient_dirs)}")
    print(f"Normal indexed folders:        {sum(1 for r in records if r['name_type'] == 'NORMAL')}")
    print(f"Anomalous folders:             {len(anomalous)}")
    print(f"Unique indices found:          {len(found_indices)}")
    print(f"Missing indices:               {len(missing_indices)}")
    print(f"Duplicate indices:             {len(duplicate_indices)}")
    print(f"Empty patient folders:         {len(empty_folders)}")
    print(f"Empty DICOM folders:           {len(empty_dicom)}")
    print(f"Missing DICOM directories:     {len(missing_dicom)}")
    print(f"Read/access errors:            {len(errors)}")

    print()
    print("Audit completed. No files or folders were modified.")


if __name__ == "__main__":
    main()
