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

EXPECTED_MISSING_INDEX = 102

OUTPUT_CSV = os.path.expanduser(
    "~/echofocus/ventricle_project/dicom_patient_metadata.csv"
)

NORMAL_PATTERN = re.compile(r"^disk_.*_(\d+)$")


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    """Convert a DICOM value to a clean string."""
    if value is None:
        return ""

    try:
        return str(value).strip()
    except Exception:
        return ""


def is_hidden(name):
    return name.startswith(".")


def extract_index(folder_name):
    """
    Extract final numeric index from normal disk_* folder names.

    Example:
        disk_20260212_192059964_99_99_398 -> 398
    """
    match = NORMAL_PATTERN.match(folder_name)

    if match:
        return int(match.group(1))

    return None


def is_real_dicom_file(filename):
    """
    Ignore AppleDouble / hidden files.
    We do not rely exclusively on the .dcm extension because
    some datasets may use different extensions.
    """

    if is_hidden(filename):
        return False

    return True


def read_dicom_metadata(path):
    """
    Read only DICOM metadata.

    stop_before_pixels=True is critical:
    PixelData is NOT loaded.
    """

    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=False
        )

        return ds, ""

    except Exception as e:
        return None, repr(e)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DICOM METADATA SEARCH FOR MISSING PATIENT INDEX 102")
    print("=" * 70)
    print()

    print("Root:")
    print(ROOT)
    print()

    if not os.path.isdir(ROOT):
        print("ERROR: Root directory is not accessible.")
        return

    # --------------------------------------------------------
    # Find patient directories
    # --------------------------------------------------------

    try:
        patient_folders = [
            name
            for name in os.listdir(ROOT)
            if not is_hidden(name)
            and os.path.isdir(os.path.join(ROOT, name))
        ]
    except Exception as e:
        print(f"ERROR reading root directory: {e}")
        return

    patient_folders.sort()

    print(f"Patient folders found: {len(patient_folders)}")
    print()

    # --------------------------------------------------------
    # Identify special folders
    # --------------------------------------------------------

    indexed_folders = {}
    anomalous_folders = []

    for folder in patient_folders:

        index = extract_index(folder)

        if index is None:
            anomalous_folders.append(folder)
        else:
            indexed_folders[index] = folder

    print(f"Normal indexed folders: {len(indexed_folders)}")
    print(f"Anomalous folders:       {len(anomalous_folders)}")
    print()

    if anomalous_folders:
        print("Anomalous folders:")
        for folder in anomalous_folders:
            print(f"  - {folder}")
        print()

    # --------------------------------------------------------
    # Focus folders
    # --------------------------------------------------------

    focus_indices = [101, 103]

    print("=" * 70)
    print("REFERENCE FOLDERS")
    print("=" * 70)

    for index in focus_indices:

        folder = indexed_folders.get(index)

        if folder:
            print(f"{index}: {folder}")
        else:
            print(f"{index}: NOT FOUND")

    print()

    print("Anomalous folders:")
    for folder in anomalous_folders:
        print(f"  - {folder}")

    print()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    records = []

    patient_metadata = defaultdict(list)

    total_dicom = 0
    total_success = 0
    total_failed = 0

    # --------------------------------------------------------
    # Scan all patients
    # --------------------------------------------------------

    for patient_number, folder in enumerate(patient_folders, start=1):

        folder_path = os.path.join(ROOT, folder)

        index = extract_index(folder)

        print(
            f"[{patient_number}/{len(patient_folders)}] "
            f"{folder}"
        )

        for root, dirs, files in os.walk(folder_path):

            # Ignore hidden directories
            dirs[:] = [
                d for d in dirs
                if not is_hidden(d)
            ]

            for filename in files:

                if not is_real_dicom_file(filename):
                    continue

                path = os.path.join(root, filename)

                total_dicom += 1

                ds, error = read_dicom_metadata(path)

                if ds is None:

                    total_failed += 1

                    records.append({
                        "folder": folder,
                        "index": index if index is not None else "",
                        "file": path,
                        "PatientID": "",
                        "PatientName": "",
                        "StudyInstanceUID": "",
                        "SeriesInstanceUID": "",
                        "SOPInstanceUID": "",
                        "StudyDate": "",
                        "StudyTime": "",
                        "Modality": "",
                        "StudyDescription": "",
                        "SeriesDescription": "",
                        "SeriesNumber": "",
                        "InstanceNumber": "",
                        "NumberOfFrames": "",
                        "Rows": "",
                        "Columns": "",
                        "ImageType": "",
                        "FrameTime": "",
                        "CineRate": "",
                        "error": error,
                    })

                    continue

                total_success += 1

                patient_id = clean(getattr(ds, "PatientID", ""))
                patient_name = clean(getattr(ds, "PatientName", ""))

                study_uid = clean(
                    getattr(ds, "StudyInstanceUID", "")
                )

                series_uid = clean(
                    getattr(ds, "SeriesInstanceUID", "")
                )

                sop_uid = clean(
                    getattr(ds, "SOPInstanceUID", "")
                )

                study_date = clean(
                    getattr(ds, "StudyDate", "")
                )

                study_time = clean(
                    getattr(ds, "StudyTime", "")
                )

                modality = clean(
                    getattr(ds, "Modality", "")
                )

                study_description = clean(
                    getattr(ds, "StudyDescription", "")
                )

                series_description = clean(
                    getattr(ds, "SeriesDescription", "")
                )

                series_number = clean(
                    getattr(ds, "SeriesNumber", "")
                )

                instance_number = clean(
                    getattr(ds, "InstanceNumber", "")
                )

                number_of_frames = clean(
                    getattr(ds, "NumberOfFrames", "")
                )

                rows = clean(
                    getattr(ds, "Rows", "")
                )

                columns = clean(
                    getattr(ds, "Columns", "")
                )

                image_type = clean(
                    getattr(ds, "ImageType", "")
                )

                frame_time = clean(
                    getattr(ds, "FrameTime", "")
                )

                cine_rate = clean(
                    getattr(ds, "CineRate", "")
                )

                record = {
                    "folder": folder,
                    "index": index if index is not None else "",
                    "file": path,
                    "PatientID": patient_id,
                    "PatientName": patient_name,
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "SOPInstanceUID": sop_uid,
                    "StudyDate": study_date,
                    "StudyTime": study_time,
                    "Modality": modality,
                    "StudyDescription": study_description,
                    "SeriesDescription": series_description,
                    "SeriesNumber": series_number,
                    "InstanceNumber": instance_number,
                    "NumberOfFrames": number_of_frames,
                    "Rows": rows,
                    "Columns": columns,
                    "ImageType": image_type,
                    "FrameTime": frame_time,
                    "CineRate": cine_rate,
                    "error": "",
                }

                records.append(record)

                # Group metadata by PatientID
                if patient_id:
                    patient_metadata[patient_id].append(record)

    # ========================================================
    # SAVE CSV
    # ========================================================

    print()
    print("=" * 70)
    print("WRITING METADATA CSV")
    print("=" * 70)

    fieldnames = [
        "folder",
        "index",
        "file",
        "PatientID",
        "PatientName",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "StudyDate",
        "StudyTime",
        "Modality",
        "StudyDescription",
        "SeriesDescription",
        "SeriesNumber",
        "InstanceNumber",
        "NumberOfFrames",
        "Rows",
        "Columns",
        "ImageType",
        "FrameTime",
        "CineRate",
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

    print(f"CSV:")
    print(OUTPUT_CSV)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("SCAN SUMMARY")
    print("=" * 70)

    print(f"Patient folders:       {len(patient_folders)}")
    print(f"DICOM files examined:  {total_dicom}")
    print(f"Metadata read OK:      {total_success}")
    print(f"Metadata read failed:  {total_failed}")
    print(f"Unique PatientIDs:     {len(patient_metadata)}")

    # ========================================================
    # PRINT METADATA FOR 101 AND 103
    # ========================================================

    print()
    print("=" * 70)
    print("PATIENT 101 / 103 METADATA")
    print("=" * 70)

    for index in focus_indices:

        folder = indexed_folders.get(index)

        if not folder:
            continue

        matching = [
            r for r in records
            if r["folder"] == folder
            and not r["error"]
        ]

        patient_ids = sorted(
            set(r["PatientID"] for r in matching if r["PatientID"])
        )

        patient_names = sorted(
            set(r["PatientName"] for r in matching if r["PatientName"])
        )

        study_uids = sorted(
            set(
                r["StudyInstanceUID"]
                for r in matching
                if r["StudyInstanceUID"]
            )
        )

        print()
        print(f"INDEX {index}")
        print(f"Folder: {folder}")
        print(f"DICOM files: {len(matching)}")
        print(f"PatientID(s): {patient_ids}")
        print(f"PatientName(s): {patient_names}")
        print(f"StudyInstanceUID(s): {study_uids}")

    # ========================================================
    # ANOMALOUS FOLDER METADATA
    # ========================================================

    print()
    print("=" * 70)
    print("ANOMALOUS FOLDER METADATA")
    print("=" * 70)

    for folder in anomalous_folders:

        matching = [
            r for r in records
            if r["folder"] == folder
            and not r["error"]
        ]

        patient_ids = sorted(
            set(r["PatientID"] for r in matching if r["PatientID"])
        )

        patient_names = sorted(
            set(r["PatientName"] for r in matching if r["PatientName"])
        )

        study_uids = sorted(
            set(
                r["StudyInstanceUID"]
                for r in matching
                if r["StudyInstanceUID"]
            )
        )

        print()
        print(f"Folder: {folder}")
        print(f"DICOM files: {len(matching)}")
        print(f"PatientID(s): {patient_ids}")
        print(f"PatientName(s): {patient_names}")
        print(f"StudyInstanceUID(s): {study_uids}")

    # ========================================================
    # PATIENT ID → FOLDER MAPPING
    # ========================================================

    print()
    print("=" * 70)
    print("PATIENTID MAPPING")
    print("=" * 70)

    patientid_to_folders = defaultdict(set)

    for r in records:

        if r["PatientID"]:
            patientid_to_folders[r["PatientID"]].add(
                r["folder"]
            )

    for patient_id in sorted(patientid_to_folders):

        folders = sorted(patientid_to_folders[patient_id])

        if len(folders) > 1:

            print()
            print(f"PatientID shared by multiple folders:")
            print(f"  PatientID: {patient_id}")

            for folder in folders:
                print(f"    - {folder}")

    # ========================================================
    # SEARCH FOR POTENTIAL 102 REFERENCES
    # ========================================================

    print()
    print("=" * 70)
    print("SEARCH FOR POSSIBLE REFERENCES TO INDEX 102")
    print("=" * 70)

    keywords = [
        "102",
        "0102",
        "00102",
    ]

    candidate_records = []

    for r in records:

        text = " ".join([
            r["PatientID"],
            r["PatientName"],
            r["StudyDescription"],
            r["SeriesDescription"],
            r["file"],
        ]).lower()

        if any(k in text for k in keywords):
            candidate_records.append(r)

    if candidate_records:

        print(
            f"Found {len(candidate_records)} metadata/file-name "
            f"records containing 102-like text."
        )

        # Only show a limited number on screen.
        shown = set()

        for r in candidate_records:

            key = (
                r["folder"],
                r["PatientID"],
                r["StudyInstanceUID"]
            )

            if key in shown:
                continue

            shown.add(key)

            print()
            print(f"Folder:      {r['folder']}")
            print(f"Index:       {r['index']}")
            print(f"PatientID:   {r['PatientID']}")
            print(f"PatientName: {r['PatientName']}")
            print(f"Study UID:   {r['StudyInstanceUID']}")

            if len(shown) >= 30:
                print()
                print("(Output limited to 30 candidate groups.)")
                break

    else:
        print("No obvious metadata/file-name reference to 102 found.")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("No files or folders were modified.")


if __name__ == "__main__":
    main()
