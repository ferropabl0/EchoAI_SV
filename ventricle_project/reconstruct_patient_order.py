#!/usr/bin/env python3

import os
import re
import csv
from collections import defaultdict

import pydicom


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"

EXPECTED_MIN = 1
EXPECTED_MAX = 398

ANOMALOUS_FOLDER = "bfr2loc_20260213_070102286_191_191"

OUTPUT_CSV = os.path.expanduser(
    "~/echofocus/ventricle_project/patient_order_reconstruction.csv"
)

# Expected normal folder:
#
# disk_20260211_223654715_19_19_101
#
# The final number is the folder index.
NORMAL_PATTERN = re.compile(
    r"^disk_(\d{8})_(\d{9})_(\d+)_(\d+)_(\d+)$"
)


# ============================================================
# HELPERS
# ============================================================

def is_hidden(name):
    return name.startswith(".")


def extract_folder_information(folder_name):
    """
    Extract information from normal disk_* folder names.

    Example:
        disk_20260211_223654715_19_19_101

    Returns:
        date
        time
        internal_id_1
        internal_id_2
        folder_index
    """

    match = NORMAL_PATTERN.match(folder_name)

    if not match:
        return {
            "date": "",
            "time": "",
            "internal_id_1": "",
            "internal_id_2": "",
            "folder_index": None,
            "normal_name": False,
        }

    return {
        "date": match.group(1),
        "time": match.group(2),
        "internal_id_1": int(match.group(3)),
        "internal_id_2": int(match.group(4)),
        "folder_index": int(match.group(5)),
        "normal_name": True,
    }


def find_first_real_dicom(folder_path):
    """
    Find the first non-hidden file that can be read as DICOM
    without loading PixelData.
    """

    for root, dirs, files in os.walk(folder_path):

        dirs[:] = [
            d for d in dirs
            if not is_hidden(d)
        ]

        for filename in sorted(files):

            if is_hidden(filename):
                continue

            path = os.path.join(root, filename)

            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=False
                )

                return ds, path

            except Exception:
                continue

    return None, None


def get_dicom_metadata(folder_path):

    ds, path = find_first_real_dicom(folder_path)

    if ds is None:
        return {
            "dicom_file": "",
            "PatientID": "",
            "PatientName": "",
            "StudyInstanceUID": "",
            "StudyDate": "",
            "StudyTime": "",
            "Modality": "",
            "error": "No readable DICOM found",
        }

    def get(tag):
        value = getattr(ds, tag, "")
        try:
            return str(value).strip()
        except Exception:
            return ""

    return {
        "dicom_file": path,
        "PatientID": get("PatientID"),
        "PatientName": get("PatientName"),
        "StudyInstanceUID": get("StudyInstanceUID"),
        "StudyDate": get("StudyDate"),
        "StudyTime": get("StudyTime"),
        "Modality": get("Modality"),
        "error": "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("PATIENT ORDER / INDEX RECONSTRUCTION")
    print("=" * 75)
    print()

    print("Root:")
    print(ROOT)
    print()

    if not os.path.isdir(ROOT):
        print("ERROR: Root directory does not exist or is inaccessible.")
        return

    # --------------------------------------------------------
    # Get top-level directories
    # --------------------------------------------------------

    folders = []

    try:

        for name in os.listdir(ROOT):

            if is_hidden(name):
                continue

            path = os.path.join(ROOT, name)

            if os.path.isdir(path):
                folders.append(name)

    except Exception as e:

        print(f"ERROR reading root directory: {e}")
        return

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # os.listdir() order is NOT guaranteed.
    #
    # We reconstruct the order using the timestamp embedded
    # in the folder name for normal disk_* folders.
    #
    # For the anomalous bfr2loc folder, we extract its
    # timestamp separately.
    # --------------------------------------------------------

    def sort_key(folder):

        info = extract_folder_information(folder)

        if info["normal_name"]:

            return (
                info["date"],
                info["time"],
                folder,
            )

        # bfr2loc_20260213_070102286_191_191
        anomalous_match = re.match(
            r"^bfr2loc_(\d{8})_(\d{9})_(\d+)_(\d+)$",
            folder
        )

        if anomalous_match:

            return (
                anomalous_match.group(1),
                anomalous_match.group(2),
                folder,
            )

        # Unknown naming format.
        return (
            "99999999",
            "999999999",
            folder,
        )

    folders.sort(key=sort_key)

    print(f"Total patient folders: {len(folders)}")
    print()

    # ========================================================
    # BUILD COMPLETE TABLE
    # ========================================================

    records = []

    for position, folder in enumerate(folders, start=1):

        info = extract_folder_information(folder)

        dicom = get_dicom_metadata(
            os.path.join(ROOT, folder)
        )

        record = {
            "position": position,
            "folder": folder,

            "folder_index": (
                info["folder_index"]
                if info["folder_index"] is not None
                else ""
            ),

            "date_from_folder": info["date"],
            "time_from_folder": info["time"],

            "internal_id_1": info["internal_id_1"],
            "internal_id_2": info["internal_id_2"],

            "normal_name": info["normal_name"],

            "PatientID": dicom["PatientID"],
            "PatientName": dicom["PatientName"],
            "StudyInstanceUID": dicom["StudyInstanceUID"],
            "StudyDate": dicom["StudyDate"],
            "StudyTime": dicom["StudyTime"],
            "Modality": dicom["Modality"],

            "dicom_file": dicom["dicom_file"],
            "error": dicom["error"],
        }

        records.append(record)

        print(
            f"{position:3d} | "
            f"index={str(record['folder_index']):>3} | "
            f"PatientID={record['PatientID']:<5} | "
            f"{folder}"
        )

    # ========================================================
    # WRITE CSV
    # ========================================================

    print()
    print("=" * 75)
    print("WRITING COMPLETE RECONSTRUCTION CSV")
    print("=" * 75)

    fieldnames = [
        "position",
        "folder",
        "folder_index",
        "date_from_folder",
        "time_from_folder",
        "internal_id_1",
        "internal_id_2",
        "normal_name",
        "PatientID",
        "PatientName",
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
        "Modality",
        "dicom_file",
        "error",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"CSV written to:")
    print(OUTPUT_CSV)

    # ========================================================
    # INDEX ANALYSIS
    # ========================================================

    index_to_records = defaultdict(list)

    for r in records:

        if r["folder_index"] != "":
            index_to_records[
                int(r["folder_index"])
            ].append(r)

    expected = set(
        range(
            EXPECTED_MIN,
            EXPECTED_MAX + 1
        )
    )

    found = set(index_to_records.keys())

    missing = sorted(expected - found)

    duplicates = {
        index: rows
        for index, rows in index_to_records.items()
        if len(rows) > 1
    }

    # ========================================================
    # FIND POSITION OF INDEX 101 / 103
    # ========================================================

    print()
    print("=" * 75)
    print("NEIGHBOURHOOD AROUND MISSING INDEX 102")
    print("=" * 75)

    relevant = [
        r for r in records
        if (
            r["folder_index"] in (101, 103)
            or r["folder"] == ANOMALOUS_FOLDER
            or abs(r["position"] - 102) <= 3
        )
    ]

    # Avoid duplicate display
    seen = set()

    for r in relevant:

        if r["position"] in seen:
            continue

        seen.add(r["position"])

        print(
            f"POSITION {r['position']:3d} | "
            f"INDEX {str(r['folder_index']):>3} | "
            f"PatientID {r['PatientID']:<5} | "
            f"{r['folder']}"
        )

    # ========================================================
    # SPECIFIC INDEX 101
    # ========================================================

    print()
    print("=" * 75)
    print("INDEX 101")
    print("=" * 75)

    for r in records:

        if r["folder_index"] == 101:

            print(f"Position:       {r['position']}")
            print(f"Folder:         {r['folder']}")
            print(f"PatientID:      {r['PatientID']}")
            print(f"PatientName:    {r['PatientName']}")
            print(f"Internal ID 1:  {r['internal_id_1']}")
            print(f"Internal ID 2:  {r['internal_id_2']}")
            print(f"Folder date:    {r['date_from_folder']}")
            print(f"Folder time:    {r['time_from_folder']}")

    # ========================================================
    # SPECIFIC INDEX 103
    # ========================================================

    print()
    print("=" * 75)
    print("INDEX 103")
    print("=" * 75)

    for r in records:

        if r["folder_index"] == 103:

            print(f"Position:       {r['position']}")
            print(f"Folder:         {r['folder']}")
            print(f"PatientID:      {r['PatientID']}")
            print(f"PatientName:    {r['PatientName']}")
            print(f"Internal ID 1:  {r['internal_id_1']}")
            print(f"Internal ID 2:  {r['internal_id_2']}")
            print(f"Folder date:    {r['date_from_folder']}")
            print(f"Folder time:    {r['time_from_folder']}")

    # ========================================================
    # ANOMALOUS FOLDER
    # ========================================================

    print()
    print("=" * 75)
    print("ANOMALOUS bfr2loc FOLDER")
    print("=" * 75)

    anomalous_records = [
        r for r in records
        if r["folder"] == ANOMALOUS_FOLDER
    ]

    if anomalous_records:

        r = anomalous_records[0]

        print(f"Position:       {r['position']}")
        print(f"Folder:         {r['folder']}")
        print(f"PatientID:      {r['PatientID']}")
        print(f"PatientName:    {r['PatientName']}")
        print(f"Internal ID 1:  {r['internal_id_1']}")
        print(f"Internal ID 2:  {r['internal_id_2']}")
        print(f"Folder date:    {r['date_from_folder']}")
        print(f"Folder time:    {r['time_from_folder']}")

    else:

        print("Anomalous folder not found.")

    # ========================================================
    # MISSING INDEX
    # ========================================================

    print()
    print("=" * 75)
    print("MISSING INDEX")
    print("=" * 75)

    print(f"Missing indices: {missing}")

    # ========================================================
    # DUPLICATES
    # ========================================================

    print()
    print("=" * 75)
    print("DUPLICATE INDICES")
    print("=" * 75)

    if duplicates:

        for index in sorted(duplicates):

            print()
            print(f"INDEX {index}:")

            for r in duplicates[index]:

                print(
                    f"  position={r['position']} "
                    f"PatientID={r['PatientID']} "
                    f"{r['folder']}"
                )

    else:

        print("No duplicate indices.")

    # ========================================================
    # POSITION / INDEX DISCREPANCIES
    # ========================================================

    print()
    print("=" * 75)
    print("POSITION vs INDEX")
    print("=" * 75)

    discrepancies = []

    for r in records:

        if r["folder_index"] == "":
            continue

        index = int(r["folder_index"])

        if index != r["position"]:

            discrepancies.append(r)

    print(
        f"Folders where position != folder index: "
        f"{len(discrepancies)}"
    )

    print()
    print("First 30 discrepancies:")

    for r in discrepancies[:30]:

        print(
            f"position={r['position']:3d} "
            f"index={int(r['folder_index']):3d} "
            f"PatientID={r['PatientID']:<5} "
            f"{r['folder']}"
        )

    if len(discrepancies) > 30:
        print(
            f"... {len(discrepancies) - 30} more. "
            f"See CSV."
        )

    # ========================================================
    # PATIENTID / INDEX RELATION
    # ========================================================

    print()
    print("=" * 75)
    print("PATIENTID vs FOLDER INDEX")
    print("=" * 75)

    same = 0
    different = 0

    for r in records:

        if r["folder_index"] == "":
            continue

        try:
            index = int(r["folder_index"])
            patient_id = int(r["PatientID"])

            if index == patient_id:
                same += 1
            else:
                different += 1

        except (ValueError, TypeError):
            pass

    print(f"Folder index == PatientID: {same}")
    print(f"Folder index != PatientID: {different}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(f"Physical patient folders:     {len(records)}")
    print(f"Expected indices:             {EXPECTED_MAX}")
    print(f"Unique indices found:         {len(found)}")
    print(f"Missing indices:              {missing}")
    print(f"Duplicate indices:            {len(duplicates)}")
    print(f"Anomalous folder:             {ANOMALOUS_FOLDER}")
    print(f"Position of anomalous folder: ", end="")

    if anomalous_records:
        print(anomalous_records[0]["position"])
    else:
        print("NOT FOUND")

    print()
    print("No files or folders were modified.")
    print()
    print("Complete CSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
