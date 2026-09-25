import os
import csv
import pydicom

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/dicom_audit.csv"
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)


def is_real_patient_folder(name):
    path = os.path.join(SRC, name)
    return os.path.isdir(path) and not name.startswith(".")


def audit_dicom(path):
    result = {
        "readable": False,
        "modality": "",
        "sop_class": "",
        "transfer_syntax": "",
        "n_frames": "",
        "rows": "",
        "columns": "",
        "photometric": "",
        "manufacturer": "",
        "model": "",
        "error": "",
    }

    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=False,
        )

        result["readable"] = True
        result["modality"] = str(getattr(ds, "Modality", ""))
        result["sop_class"] = str(
            getattr(ds, "SOPClassUID", "")
        )
        result["transfer_syntax"] = str(
            getattr(ds.file_meta, "TransferSyntaxUID", "")
        )
        result["n_frames"] = int(getattr(ds, "NumberOfFrames", 1))
        result["rows"] = int(getattr(ds, "Rows", 0))
        result["columns"] = int(getattr(ds, "Columns", 0))
        result["photometric"] = str(
            getattr(ds, "PhotometricInterpretation", "")
        )
        result["manufacturer"] = str(
            getattr(ds, "Manufacturer", "")
        )
        result["model"] = str(
            getattr(ds, "ManufacturerModelName", "")
        )

    except Exception as e:
        result["error"] = type(e).__name__ + ": " + str(e)[:200]

    return result


print("Scanning patient folders...")

patient_folders = [
    name
    for name in os.listdir(SRC)
    if is_real_patient_folder(name)
]

print(f"Patient folders found: {len(patient_folders)}")

rows = []

for patient_index, patient in enumerate(patient_folders):
    patient_path = os.path.join(SRC, patient)
    dicom_dir = os.path.join(patient_path, "DICOM")

    result = {
        "patient_index": patient_index,
        "has_dicom_folder": os.path.isdir(dicom_dir),
        "n_dicom_files": 0,
        "n_readable": 0,
        "n_multiframe": 0,
        "n_singleframe": 0,
        "n_us": 0,
        "n_valid_multiframe": 0,
        "total_frames": 0,
        "min_frames": "",
        "max_frames": "",
        "common_rows": "",
        "common_columns": "",
        "errors": 0,
    }

    if not os.path.isdir(dicom_dir):
        rows.append(result)
        continue

    dicom_files = []

    for filename in os.listdir(dicom_dir):
        if filename.startswith("."):
            continue

        path = os.path.join(dicom_dir, filename)

        if os.path.isfile(path) and filename.lower().endswith(".dcm"):
            dicom_files.append(path)

    result["n_dicom_files"] = len(dicom_files)

    frame_counts = []
    resolutions = []

    for path in dicom_files:
        info = audit_dicom(path)

        if not info["readable"]:
            result["errors"] += 1
            continue

        result["n_readable"] += 1

        if info["modality"] == "US":
            result["n_us"] += 1

        n_frames = info["n_frames"]

        if n_frames > 1:
            result["n_multiframe"] += 1
            frame_counts.append(n_frames)

            if (
                info["modality"] == "US"
                and info["rows"] > 0
                and info["columns"] > 0
            ):
                result["n_valid_multiframe"] += 1
                resolutions.append(
                    f'{info["columns"]}x{info["rows"]}'
                )
        else:
            result["n_singleframe"] += 1

    if frame_counts:
        result["total_frames"] = sum(frame_counts)
        result["min_frames"] = min(frame_counts)
        result["max_frames"] = max(frame_counts)

    if resolutions:
        from collections import Counter

        result["common_rows"] = Counter(resolutions).most_common(3)

    rows.append(result)

    if (patient_index + 1) % 25 == 0:
        print(f"Processed {patient_index + 1}/{len(patient_folders)}")


fieldnames = list(rows[0].keys()) if rows else []

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print()
print("========================================")
print("AUDIT COMPLETE")
print("========================================")
print(f"Patients: {len(rows)}")
print(f"Patients with DICOM folder: {sum(r['has_dicom_folder'] for r in rows)}")
print(f"Patients with multiframe DICOMs: {sum(r['n_multiframe'] > 0 for r in rows)}")
print(f"Patients with valid US multiframe: {sum(r['n_valid_multiframe'] > 0 for r in rows)}")
print(f"Total DICOM files: {sum(r['n_dicom_files'] for r in rows)}")
print(f"Total readable DICOMs: {sum(r['n_readable'] for r in rows)}")
print(f"Total multiframe DICOMs: {sum(r['n_multiframe'] for r in rows)}")
print(f"Total valid US multiframe: {sum(r['n_valid_multiframe'] for r in rows)}")
print()
print(f"Saved to: {OUT}")
