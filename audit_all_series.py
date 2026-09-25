import os
import csv
from collections import defaultdict
import pydicom

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/all_series_audit.csv"
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def discover_patients():
    return sorted(
        d for d in os.listdir(SRC)
        if not d.startswith(".")
        and os.path.isdir(os.path.join(SRC, d))
    )


patients = discover_patients()

print("=" * 90)
print("FULL DICOM SERIES AUDIT")
print("=" * 90)
print("Patients:", len(patients))
print()


rows = []

for patient_idx, patient in enumerate(patients, 1):

    patient_dir = os.path.join(SRC, patient, "DICOM")

    if not os.path.isdir(patient_dir):
        rows.append({
            "patient": patient,
            "category": "no_DICOM_folder",
        })
        continue

    # --------------------------------------------------------
    # Find real DICOM candidates recursively
    # --------------------------------------------------------

    files = []

    for root, _, filenames in os.walk(patient_dir):

        for name in filenames:

            # Ignore macOS metadata
            if name.startswith("."):
                continue

            if not name.lower().endswith(".dcm"):
                continue

            files.append(os.path.join(root, name))

    # --------------------------------------------------------
    # Group by SeriesInstanceUID
    # --------------------------------------------------------

    series = defaultdict(list)
    errors = []

    for path in files:

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )

            modality = str(
                getattr(ds, "Modality", "")
            )

            series_uid = str(
                getattr(ds, "SeriesInstanceUID", "")
            )

            number_of_frames = safe_int(
                getattr(ds, "NumberOfFrames", None)
            )

            instance = safe_int(
                getattr(ds, "InstanceNumber", None)
            )

            frame_time = safe_float(
                getattr(ds, "FrameTime", None)
            )

            cine_rate = safe_float(
                getattr(ds, "CineRate", None)
            )

            recommended_rate = safe_float(
                getattr(
                    ds,
                    "RecommendedDisplayFrameRate",
                    None
                )
            )

            image_type = getattr(
                ds,
                "ImageType",
                None
            )

            if image_type is None:
                image_type_str = ""
            else:
                image_type_str = "\\".join(
                    str(x) for x in image_type
                )

            rows.append({
                "patient": patient,
                "category": "series_file",
                "series_uid": series_uid,
                "modality": modality,
                "path": path,
                "number_of_frames": number_of_frames,
                "instance_number": instance,
                "frame_time_ms": frame_time,
                "cine_rate": cine_rate,
                "recommended_display_frame_rate": recommended_rate,
                "image_type": image_type_str,
                "rows": safe_int(getattr(ds, "Rows", None)),
                "columns": safe_int(getattr(ds, "Columns", None)),
            })

            series[series_uid].append({
                "instance": instance,
                "frames": number_of_frames,
                "frame_time": frame_time,
                "cine_rate": cine_rate,
                "recommended_rate": recommended_rate,
                "image_type": image_type_str,
                "modality": modality,
            })

        except Exception as e:
            errors.append({
                "path": path,
                "error": repr(e)
            })

    # --------------------------------------------------------
    # Patient summary
    # --------------------------------------------------------

    us_series = {
        uid: items
        for uid, items in series.items()
        if any(
            x["modality"] == "US"
            for x in items
        )
    }

    multiframe_series = {}
    singleframe_cine_series = {}
    singleframe_static_series = {}

    for uid, items in us_series.items():

        # True multiframe DICOM
        if any(
            x["frames"] is not None and x["frames"] > 1
            for x in items
        ):
            multiframe_series[uid] = items
            continue

        # Single-frame series
        instances = [
            x["instance"]
            for x in items
            if x["instance"] is not None
        ]

        instances_sorted = sorted(instances)

        consecutive = (
            len(instances_sorted) >= 2
            and instances_sorted == list(
                range(
                    instances_sorted[0],
                    instances_sorted[-1] + 1
                )
            )
        )

        frame_times = [
            x["frame_time"]
            for x in items
            if x["frame_time"] is not None
        ]

        has_temporal_info = len(frame_times) > 0

        image_type_text = " ".join(
            x["image_type"]
            for x in items
        )

        says_multiframe = (
            "GEMSMULTIFRAME" in image_type_text
        )

        if (
            len(items) >= 16
            and consecutive
            and has_temporal_info
        ) or says_multiframe:

            singleframe_cine_series[uid] = items

        else:
            singleframe_static_series[uid] = items

    # --------------------------------------------------------
    # Determine patient category
    # --------------------------------------------------------

    if multiframe_series and singleframe_cine_series:
        category = "mixed"

    elif multiframe_series:
        category = "multiframe_US"

    elif singleframe_cine_series:
        category = "singleframe_cine"

    elif singleframe_static_series:
        category = "singleframe_static"

    elif us_series:
        category = "US_other"

    elif files:
        category = "non_US"

    else:
        category = "no_DICOM"

    # --------------------------------------------------------
    # Print useful summary
    # --------------------------------------------------------

    print(
        f"[{patient_idx:3d}/{len(patients)}] "
        f"{patient[:45]:45s} "
        f"{category:20s} "
        f"files={len(files):4d} "
        f"US_series={len(us_series):2d} "
        f"cine={len(singleframe_cine_series):2d} "
        f"multi={len(multiframe_series):2d}"
    )

    # Add one patient-level summary row
    rows.append({
        "patient": patient,
        "category": "PATIENT_SUMMARY",
        "series_uid": "",
        "modality": "US" if us_series else "",
        "path": "",
        "number_of_frames": "",
        "instance_number": "",
        "frame_time_ms": "",
        "cine_rate": "",
        "recommended_display_frame_rate": "",
        "image_type": "",
        "rows": "",
        "columns": "",
    })


# ------------------------------------------------------------
# Write CSV
# ------------------------------------------------------------

fieldnames = [
    "patient",
    "category",
    "series_uid",
    "modality",
    "path",
    "number_of_frames",
    "instance_number",
    "frame_time_ms",
    "cine_rate",
    "recommended_display_frame_rate",
    "image_type",
    "rows",
    "columns",
]

with open(
    OUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)


# ------------------------------------------------------------
# Final statistics
# ------------------------------------------------------------

print()
print("=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)
print("Output:", OUT)

# Recalculate patient categories from the series information
# stored in the CSV is intentionally avoided here; the CSV is
# the detailed output for the next analysis step.

print()
print("The CSV contains:")
print("  - one row per DICOM file")
print("  - patient summary markers")
print("  - SeriesInstanceUID")
print("  - InstanceNumber")
print("  - FrameTime")
print("  - CineRate")
print("  - ImageType")
print("  - dimensions")
